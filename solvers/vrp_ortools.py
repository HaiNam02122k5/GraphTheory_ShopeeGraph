"""
VRPOrToolsSolver — Rolling-Horizon VRP cho Online MAPD.

Kiến trúc:
  1. BFS distance/path cache         — tính khoảng cách & đường đi thực trên grid
  2. Distance matrix builder         — snapshot node set → ma trận khoảng cách
  3. VRP model (OR-Tools)            — Pickup-Delivery + Time Window + Capacity
  4. Route Executor                  — dịch route → deque(moves), thực thi từng bước
  5. Rolling-Horizon re-planner      — trigger khi có đơn mới / định kỳ Δt bước
  6. Greedy fallback                 — khi OR-Tools timeout hoặc infeasible

Complexity:
  - Distance matrix build: O(P × N²) với P = số node, N = cạnh grid
  - OR-Tools VRP: NP-hard nhưng bounded bởi time_limit
  - Mỗi bước: O(1) amortized (pop từ deque)
"""
from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, valid_next_pos
from solvers.solver import Solver

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Position = Tuple[int, int]
Move = str
Action = Tuple[Move, int]
RouteStop = Tuple[Position, int, int]  # (target position, cargo op, order id)

INF = 10 ** 9
MOVES: Tuple[str, ...] = ("U", "D", "L", "R")

# ---------------------------------------------------------------------------
# Tunable hyper-parameters
# ---------------------------------------------------------------------------
REPLAN_INTERVAL_SMALL = 10   # Δt cho config nhỏ (N < 18)
REPLAN_INTERVAL_LARGE = 20   # Δt cho config lớn (N >= 18)
ORTOOLS_TIME_LIMIT_S  = 5    # Giới hạn thời gian OR-Tools (giây)
MAX_ACTIVE_ORDERS     = 40   # Số đơn tối đa đưa vào VRP model mỗi lần (tránh quá lớn)
MIN_EXPECTED_REWARD   = 0.5  # Bỏ đơn có expected reward < ngưỡng này
MAX_HEURISTIC_STOPS   = 8    # Số stop tối đa trong fallback planner nội bộ
DISTANCE_COST_SCALE   = 10   # Scale objective distance để cân bằng với drop penalty
DROP_REWARD_SCALE     = 35   # Reward dự kiến -> penalty nếu bỏ đơn
DROP_PRIORITY_BONUS   = 40   # Bonus penalty cho đơn priority cao
LATE_COST_SCALE       = 25   # Soft deadline penalty trên mỗi timestep trễ


from solvers.pathfinder import get_pathfinder

# ===========================================================================
# VRPOrToolsSolver
# ===========================================================================

class VRPOrToolsSolver(Solver):
    """Rolling-Horizon VRP + OR-Tools solver cho Online MAPD."""

    method_name = "VRPOrToolsSolver"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)

        # Khởi tạo PathFinder tăng tốc bởi GPU/CPU
        self.pathfinder = get_pathfinder(self.grid)

        # Kế hoạch hiện tại: shipper_id → deque of (target_pos, cargo_op)
        # cargo_op: 0=move only, 1=pickup khi đến, 2=deliver khi đến
        self._plans: Dict[int, deque] = {}

        # Đơn đã assign cho shipper nhưng chưa pickup
        self._assigned: Dict[int, int] = {}   # order_id → shipper_id

        self._last_plan_t: int = -99
        self._replan_interval = (
            REPLAN_INTERVAL_LARGE if env.N >= 18 else REPLAN_INTERVAL_SMALL
        )
        self._prev_pending_count: int = 0
        if self.env.N <= 22:
            self.max_delivery_delay = 10
        elif self.env.N <= 25:
            self.max_delivery_delay = 5
        elif self.env.N <= 30:
            self.max_delivery_delay = -5
        elif self.env.N <= 35:
            self.max_delivery_delay = -8
        else:
            self.max_delivery_delay = -10

    # -----------------------------------------------------------------------
    # BFS utilities
    # -----------------------------------------------------------------------

    def _bfs(self, start: Position, goal: Position) -> Tuple[int, List[Move]]:
        """BFS với cache."""
        return self.pathfinder.dist(start, goal), self.pathfinder.path(start, goal)

    def _dist(self, a: Position, b: Position) -> int:
        return self.pathfinder.dist(a, b)

    def _path(self, a: Position, b: Position) -> List[Move]:
        return self.pathfinder.path(a, b)

    # -----------------------------------------------------------------------
    # Expected reward helper
    # -----------------------------------------------------------------------

    def _expected_reward(self, from_pos: Position, order: Order, obs_t: int) -> float:
        d1 = self._dist(from_pos, (order.sx, order.sy))
        d2 = self._dist((order.sx, order.sy), (order.ex, order.ey))
        return delivery_reward(order, obs_t + d1 + d2, self.env.T)

    def _pickup_score(
        self,
        from_pos: Position,
        order: Order,
        obs_t: int,
        elapsed: int = 0,
    ) -> float:
        d_pick = self._dist(from_pos, (order.sx, order.sy))
        d_drop = self._dist((order.sx, order.sy), (order.ex, order.ey))
        if d_pick >= INF or d_drop >= INF:
            return 0.0
        eta_delivery = obs_t + elapsed + d_pick + d_drop
        if eta_delivery - order.et > self.max_delivery_delay:
            return 0.0
        reward = delivery_reward(order, eta_delivery, self.env.T)
        urgency = 0.0
        if eta_delivery <= order.et:
            urgency = 1.0 / max(order.et - obs_t, 1)
        return reward / (d_pick + d_drop + 1) + urgency * 10.0 + 0.05 * order.p

    def _select_delivery_order(self, shipper: Shipper, orders: Dict[int, Order]) -> Optional[Order]:
        carried = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if not carried:
            return None

        obs_t = self.env.t
        def key(order: Order):
            d = self._dist(shipper.position, (order.ex, order.ey))
            if obs_t + d <= order.et:
                return (0, order.et, d, -order.p, order.id)
            return (1, d, order.et, -order.p, order.id)

        return min(carried, key=key)

    # -----------------------------------------------------------------------
    # Distance matrix builder
    # -----------------------------------------------------------------------

    def _build_distance_matrix(
        self, positions: List[Position]
    ) -> List[List[int]]:
        """
        Xây ma trận khoảng cách BFS giữa tất cả cặp position.
        Kết quả: dist_matrix[i][j] = distance(positions[i], positions[j])
        """
        n = len(positions)
        mat = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = self._dist(positions[i], positions[j])
                mat[i][j] = d
                mat[j][i] = d
        return mat

    # -----------------------------------------------------------------------
    # Greedy fallback (dùng khi OR-Tools thất bại)
    # -----------------------------------------------------------------------

    def _greedy_fallback(self, obs: dict) -> Dict[int, Action]:
        """
        Fallback về Greedy BFS đơn giản.
        Logic: delivery trước, pickup sau, idle = đứng yên.
        """
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        obs_t: int = obs["t"]

        actions: Dict[int, Action] = {}
        reserved: set = set()

        for s in sorted(shippers, key=lambda s: s.id):
            # Giao đơn đang mang
            carried = [
                orders[oid]
                for oid in s.bag
                if oid in orders and not orders[oid].delivered
            ]
            if carried:
                best = self._select_delivery_order(s, orders)
                if best is None:
                    actions[s.id] = ("S", 0)
                    continue
                goal = (best.ex, best.ey)
                mv = self._path(s.position, goal)
                move = mv[0] if mv else "S"
                nxt = valid_next_pos(s.position, move, self.grid)
                actions[s.id] = (move, 2) if nxt == goal else (move, 0)
                continue

            # Nhặt đơn
            cands = []
            for o in orders.values():
                if o.id in reserved or not s.can_carry(o, orders):
                    continue
                d = self._dist(s.position, (o.sx, o.sy))
                if d >= INF:
                    continue
                d2 = self._dist((o.sx, o.sy), (o.ex, o.ey))
                if obs_t + d + d2 - o.et > self.max_delivery_delay:
                    continue
                score = self._pickup_score(s.position, o, obs_t)
                if score <= 0.0:
                    continue
                cands.append((score, d, o))

            if cands:
                _, _, best = max(
                    cands,
                    key=lambda item: (item[0], -item[1], -item[2].id),
                )
                reserved.add(best.id)
                goal = (best.sx, best.sy)
                mv = self._path(s.position, goal)
                move = mv[0] if mv else "S"
                nxt = valid_next_pos(s.position, move, self.grid)
                actions[s.id] = (move, 1) if nxt == goal else (move, 0)
                continue

            actions[s.id] = ("S", 0)

        return actions

    # -----------------------------------------------------------------------
    # VRP planner (OR-Tools)
    # -----------------------------------------------------------------------

    def _run_heuristic_vrp(self, obs: dict) -> Optional[Dict[int, List[RouteStop]]]:
        """
        Planner nội bộ khi OR-Tools không có sẵn.

        Đây không phải greedy từng bước: mỗi shipper xây một route nhiều stop,
        chấm điểm theo reward dự kiến / quãng đường còn lại, và giữ reservation
        cấp route để các shipper không cùng chạy tới một đơn.
        """
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        obs_t: int = obs["t"]

        unpicked = [
            o for o in orders.values()
            if not o.picked and not o.delivered
        ]
        if not unpicked and not any(s.bag for s in shippers):
            return None

        assigned: Set[int] = set()
        routes: Dict[int, List[RouteStop]] = {}

        # Shipper đang rảnh được ưu tiên lập route pickup trước; shipper đang mang
        # vẫn được phép pickup thêm nếu detour tốt và còn sức chứa.
        shipper_order = sorted(
            shippers,
            key=lambda s: (len(s.bag), s.id),
        )

        for s in shipper_order:
            pos = s.position
            elapsed = 0
            compact_mid_map = 12 <= self.env.N < 18
            route_limit = 4 if compact_mid_map else MAX_HEURISTIC_STOPS
            bag: List[int] = [
                oid for oid in s.bag
                if oid in orders and not orders[oid].delivered
            ]
            route: List[RouteStop] = []
            carried_weight = sum(orders[oid].w for oid in bag if oid in orders)

            while len(route) < route_limit:
                candidates = []

                # Delivery candidates for orders already planned/carried.
                for oid in list(bag):
                    o = orders.get(oid)
                    if o is None or o.delivered:
                        continue
                    dst = (o.ex, o.ey)
                    d = self._dist(pos, dst)
                    if d >= INF:
                        continue
                    arrival = obs_t + elapsed + d
                    reward = delivery_reward(o, arrival, self.env.T)
                    urgency = 1.0
                    if arrival > o.et:
                        urgency += min(1.0, (arrival - o.et) / max(self.env.T, 1))
                    score = (reward * urgency) / (d + 1)
                    candidates.append((score, -d, -o.p, -o.id, "deliver", o, d))

                # Pickup candidates. Score by full pickup->delivery value, but
                # favor nearby pickups enough to create useful batching.
                if len(bag) < s.K_max:
                    for o in unpicked:
                        if o.id in assigned or o.id in bag:
                            continue
                        if carried_weight + o.w > s.W_max:
                            continue
                        src = (o.sx, o.sy)
                        dst = (o.ex, o.ey)
                        d_pick = self._dist(pos, src)
                        d_drop = self._dist(src, dst)
                        if d_pick >= INF or d_drop >= INF:
                            continue
                        eta_delivery = obs_t + elapsed + d_pick + d_drop
                        reward = delivery_reward(o, eta_delivery, self.env.T)
                        if reward < MIN_EXPECTED_REWARD:
                            continue
                        lateness = max(0, eta_delivery - o.et)
                        if compact_mid_map and lateness > self.env.T * 0.25:
                            continue
                        distance_value = reward / (d_pick + d_drop + 1)
                        pickup_value = 0.35 * reward / (d_pick + 1)
                        deadline_factor = 1.0
                        if eta_delivery > o.et:
                            deadline_factor -= min(0.5, lateness / max(self.env.T, 1))
                        score = (distance_value + pickup_value) * deadline_factor + 0.05 * o.p
                        candidates.append((score, -d_pick, -o.p, -o.id, "pickup", o, d_pick))

                if not candidates:
                    break

                candidates.sort(reverse=True)
                _, _, _, _, kind, order, travel = candidates[0]
                if kind == "pickup":
                    route.append(((order.sx, order.sy), 1, order.id))
                    assigned.add(order.id)
                    bag.append(order.id)
                    carried_weight += order.w
                    pos = (order.sx, order.sy)
                    elapsed += travel
                else:
                    route.append(((order.ex, order.ey), 2, order.id))
                    if order.id in bag:
                        bag.remove(order.id)
                    carried_weight = max(0.0, carried_weight - order.w)
                    pos = (order.ex, order.ey)
                    elapsed += travel

                if elapsed >= self.env.T:
                    break

            routes[s.id] = route

        return routes if any(routes.values()) else None

    def _run_vrp(self, obs: dict) -> Optional[Dict[int, List[RouteStop]]]:
        """
        Giải VRP bằng OR-Tools.
        Trả về Dict[shipper_id → [pos0, pos1, ..., posK]] (chuỗi vị trí cần ghé)
        hoặc None nếu thất bại/timeout.
        """
        try:
            from ortools.constraint_solver import pywrapcp as pw, routing_enums_pb2
        except ImportError:
            return self._run_heuristic_vrp(obs)

        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        obs_t: int = obs["t"]
        T = self.env.T

        # --- Chọn đơn đưa vào VRP ---
        # Đơn chưa picked, còn thời gian, expected reward đủ lớn.
        # Reward dự kiến cũng được dùng làm drop penalty trong OR-Tools.
        unassigned_orders: List[Order] = []
        expected_rewards: Dict[int, float] = {}
        for o in orders.values():
            if o.picked or o.delivered:
                continue
            nearest_pos = min(
                (s.position for s in shippers),
                key=lambda p: self._dist(p, (o.sx, o.sy)),
            )
            expected = self._expected_reward(nearest_pos, o, obs_t)
            if expected < MIN_EXPECTED_REWARD:
                continue
            expected_rewards[o.id] = expected
            unassigned_orders.append(o)

        # Giới hạn số đơn để tránh model quá lớn
        unassigned_orders.sort(key=lambda o: (-expected_rewards[o.id], o.et, -o.p, o.id))
        unassigned_orders = unassigned_orders[:MAX_ACTIVE_ORDERS]

        # Đơn đang mang: phải deliver trước
        in_transit: Dict[int, List[Order]] = {s.id: [] for s in shippers}
        for s in shippers:
            for oid in s.bag:
                if oid in orders and not orders[oid].delivered:
                    in_transit[s.id].append(orders[oid])

        if not unassigned_orders and not any(in_transit.values()):
            return None

        # --- Xây node list ---
        # Node 0 = depot ảo.
        # Mỗi vehicle có start node riêng và end dummy riêng. Cost đi vào end
        # dummy bằng 0 để mô hình là open-route, không ép shipper quay về depot.
        num_vehicles = len(shippers)

        node_positions: List[Position] = []
        node_stops: Dict[int, RouteStop] = {}

        # depot ảo: dùng vị trí shipper 0 (irrelevant vì mỗi vehicle bắt đầu từ node riêng)
        depot_pos = shippers[0].position
        node_positions.append(depot_pos)  # node 0 = depot

        # start nodes: node 1..C
        start_nodes = []
        for s in shippers:
            start_nodes.append(len(node_positions))
            node_positions.append(s.position)

        end_nodes = []
        end_dummy_nodes: Set[int] = set()
        for s in shippers:
            end_node = len(node_positions)
            end_nodes.append(end_node)
            end_dummy_nodes.add(end_node)
            node_positions.append(s.position)

        # pickup + delivery nodes
        pickup_nodes: Dict[int, int] = {}   # order_id → node index
        delivery_nodes: Dict[int, int] = {} # order_id → node index
        for o in unassigned_orders:
            pickup_nodes[o.id] = len(node_positions)
            node_positions.append((o.sx, o.sy))
            node_stops[pickup_nodes[o.id]] = ((o.sx, o.sy), 1, o.id)
            delivery_nodes[o.id] = len(node_positions)
            node_positions.append((o.ex, o.ey))
            node_stops[delivery_nodes[o.id]] = ((o.ex, o.ey), 2, o.id)

        # Đơn đang mang: chỉ cần delivery node
        transit_delivery_nodes: Dict[int, int] = {}
        for s in shippers:
            for o in in_transit[s.id]:
                if o.id not in transit_delivery_nodes:
                    transit_delivery_nodes[o.id] = len(node_positions)
                    node_positions.append((o.ex, o.ey))
                    node_stops[transit_delivery_nodes[o.id]] = ((o.ex, o.ey), 2, o.id)

        num_nodes = len(node_positions)

        # --- Distance matrix ---
        dist_matrix = self._build_distance_matrix(node_positions)

        # --- OR-Tools setup ---
        manager = pw.RoutingIndexManager(num_nodes, num_vehicles, start_nodes, end_nodes)
        routing = pw.RoutingModel(manager)

        # Distance callback: dùng distance raw cho Time/Capacity precedence,
        # dùng scaled cost cho objective để cân bằng với drop/late penalties.
        def distance_callback(from_idx, to_idx):
            to_node = manager.IndexToNode(to_idx)
            if to_node in end_dummy_nodes:
                return 0
            return dist_matrix[manager.IndexToNode(from_idx)][to_node]

        transit_cb = routing.RegisterTransitCallback(distance_callback)

        def cost_callback(from_idx, to_idx):
            return distance_callback(from_idx, to_idx) * DISTANCE_COST_SCALE

        cost_cb = routing.RegisterTransitCallback(cost_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(cost_cb)

        # --- Time: precedence + soft deadline ---
        horizon = max(T - obs_t + self.env.N * self.env.N, 1)
        routing.AddDimension(transit_cb, 0, horizon, True, "Time")
        time_dim = routing.GetDimensionOrDie("Time")

        for o in unassigned_orders:
            p_idx = manager.NodeToIndex(pickup_nodes[o.id])
            d_idx = manager.NodeToIndex(delivery_nodes[o.id])
            routing.solver().Add(time_dim.CumulVar(p_idx) <= time_dim.CumulVar(d_idx))
            due = max(0, o.et - obs_t)
            time_dim.SetCumulVarSoftUpperBound(
                d_idx,
                due,
                LATE_COST_SCALE * max(1, o.p),
            )

        for o_id, d_node in transit_delivery_nodes.items():
            o = orders[o_id]
            d_idx = manager.NodeToIndex(d_node)
            due = max(0, o.et - obs_t)
            time_dim.SetCumulVarSoftUpperBound(
                d_idx,
                due,
                LATE_COST_SCALE * max(1, o.p),
            )

        # --- Capacity: Weight (UnaryTransitCallback: f(idx) -> demand) ---
        node_weight: List[int] = [0] * num_nodes
        for o in unassigned_orders:
            node_weight[pickup_nodes[o.id]]   =  int(o.w * 10)
            node_weight[delivery_nodes[o.id]] = -int(o.w * 10)
        for o_id, d_idx in transit_delivery_nodes.items():
            node_weight[d_idx] = -int(orders[o_id].w * 10)

        def weight_callback(idx):
            return node_weight[manager.IndexToNode(idx)]

        weight_cb = routing.RegisterUnaryTransitCallback(weight_callback)
        routing.AddDimensionWithVehicleCapacity(
            weight_cb,
            0,
            [int(s.W_max * 10) for s in shippers],
            False,
            "Weight",
        )
        weight_dim = routing.GetDimensionOrDie("Weight")
        for v_idx, s in enumerate(shippers):
            initial_weight = int(
                sum(orders[oid].w for oid in s.bag if oid in orders) * 10
            )
            weight_dim.CumulVar(routing.Start(v_idx)).SetValue(initial_weight)

        # --- Capacity: Slots ---
        node_slots: List[int] = [0] * num_nodes
        for o in unassigned_orders:
            node_slots[pickup_nodes[o.id]]   =  1
            node_slots[delivery_nodes[o.id]] = -1
        for d_idx in transit_delivery_nodes.values():
            node_slots[d_idx] = -1

        def slot_callback(idx):
            return node_slots[manager.IndexToNode(idx)]

        slot_cb = routing.RegisterUnaryTransitCallback(slot_callback)
        routing.AddDimensionWithVehicleCapacity(
            slot_cb,
            0,
            [s.K_max for s in shippers],
            False,
            "Slots",
        )
        slot_dim = routing.GetDimensionOrDie("Slots")
        for v_idx, s in enumerate(shippers):
            initial_slots = sum(
                1 for oid in s.bag if oid in orders and not orders[oid].delivered
            )
            slot_dim.CumulVar(routing.Start(v_idx)).SetValue(initial_slots)

        # --- Pickup-Delivery constraints ---
        for o in unassigned_orders:
            p_idx = manager.NodeToIndex(pickup_nodes[o.id])
            d_idx = manager.NodeToIndex(delivery_nodes[o.id])
            routing.AddPickupAndDelivery(p_idx, d_idx)
            routing.solver().Add(
                routing.VehicleVar(p_idx) == routing.VehicleVar(d_idx)
            )
            routing.solver().Add(routing.ActiveVar(p_idx) == routing.ActiveVar(d_idx))
            pair_penalty = int(
                max(
                    DISTANCE_COST_SCALE,
                    expected_rewards[o.id] * DROP_REWARD_SCALE
                    + o.p * DROP_PRIORITY_BONUS,
                )
            )
            routing.AddDisjunction([p_idx], pair_penalty // 2)
            routing.AddDisjunction([d_idx], pair_penalty - pair_penalty // 2)

        # --- In-transit: shipper đang mang phải deliver ---
        # (SetMin trên CumulVar capacity gây CP Solver fail → chỉ dùng VehicleVar constraint)
        for v_idx, s in enumerate(shippers):
            for o in in_transit[s.id]:
                if o.id in transit_delivery_nodes:
                    d_idx = manager.NodeToIndex(transit_delivery_nodes[o.id])
                    routing.VehicleVar(d_idx).SetValues([v_idx])

        # --- Search parameters ---
        # Dùng FIRST SOLUTION only (không GLS) để đảm bảo terminate đúng thời hạn.
        # GLS trong OR-Tools Python binding bỏ qua time_limit → hang vô tận.
        params = pw.DefaultRoutingSearchParameters()
        params.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
        )
        # Không set local_search_metaheuristic → mặc định AUTOMATIC (chỉ first solution)
        params.time_limit.seconds = ORTOOLS_TIME_LIMIT_S
        params.solution_limit = 1   # dừng ngay sau first solution
        params.log_search = False

        try:
            solution = routing.SolveWithParameters(params)
        except Exception:
            return None
        if solution is None:
            return None

        # --- Extract routes ---
        routes: Dict[int, List[RouteStop]] = {}
        for v_idx, s in enumerate(shippers):
            route_stops: List[RouteStop] = []
            idx = routing.Start(v_idx)
            while not routing.IsEnd(idx):
                node = manager.IndexToNode(idx)
                if node in node_stops:
                    route_stops.append(node_stops[node])
                idx = solution.Value(routing.NextVar(idx))
            routes[s.id] = route_stops

        return routes

    # -----------------------------------------------------------------------
    # Route → Target list (just positions, no pre-computed moves)
    # -----------------------------------------------------------------------

    def _extract_targets(
        self,
        routes: Dict[int, List[RouteStop]],
        obs: dict,
    ) -> Dict[int, deque]:
        """
        VRP route → deque[RouteStop].
        Moves sẽ được tính live mỗi bước, nhưng giữ order id/op để không mất
        nghĩa khi pickup và delivery có cùng tọa độ.
        """
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        shipper_map = {s.id: s for s in shippers}

        targets: Dict[int, deque] = {}

        for s_id, positions in routes.items():
            s = shipper_map[s_id]
            tgt: deque = deque()

            # Đơn đang mang nhưng route chưa có delivery thì thêm vào đầu.
            in_bag = [
                orders[oid] for oid in s.bag
                if oid in orders and not orders[oid].delivered
            ]
            planned_oids = {oid for _, _, oid in positions}
            for o in sorted(in_bag, key=lambda o: (self._dist(s.position, (o.ex, o.ey)), o.et)):
                if o.id not in planned_oids:
                    tgt.append(((o.ex, o.ey), 2, o.id))

            for stop in positions:
                pos, op, oid = stop
                if pos != s.position or self._is_stop_actionable(stop, s, obs):
                    tgt.append((pos, op, oid))

            targets[s_id] = tgt

        return targets

    # -----------------------------------------------------------------------
    # Step execution: navigate toward current target, determine cargo_op live
    # -----------------------------------------------------------------------

    def _is_stop_actionable(self, stop: RouteStop, s: Shipper, obs: dict) -> bool:
        """True nếu stop còn hợp lệ với observation hiện tại."""
        pos, op, oid = stop
        orders: Dict[int, Order] = obs["orders"]
        order = orders.get(oid)
        if order is None or order.delivered:
            return False
        if op == 1:
            return (
                not order.picked
                and (order.sx, order.sy) == pos
                and s.can_carry(order, orders)
            )
        if op == 2:
            return (
                oid in s.bag
                and not order.delivered
                and (order.ex, order.ey) == pos
            )
        return False

    def _planned_stop_action(self, stop: RouteStop, s: Shipper, obs: dict) -> Action:
        pos, op, _ = stop
        return self._navigate_to(s.position, pos, cargo_op_at_goal=op)

    def _step_action(self, s: Shipper, obs: dict) -> Action:
        """
        Tính action cho 1 shipper tại bước hiện tại.
        Dùng target list nếu có, fallback greedy nếu không.
        """
        orders: Dict[int, Order] = obs["orders"]

        # --- Priority 1: Follow planned VRP targets ---
        tgt_queue = self._targets.get(s.id)
        if tgt_queue:
            # Bỏ stop đã hết hợp lệ do shipper khác đã pickup, đã giao, hoặc
            # capacity hiện tại không còn phù hợp.
            while tgt_queue and not self._is_stop_actionable(tgt_queue[0], s, obs):
                tgt_queue.popleft()

            if tgt_queue:
                stop = tgt_queue[0]
                goal, op, _ = stop
                vrp_dist = self._dist(s.position, goal)

                # Với delivery đang mang, tôn trọng route vì đó là ràng buộc thực.
                if op == 2:
                    return self._planned_stop_action(stop, s, obs)

                # Với pickup, nếu route quá xa so với cơ hội live gần nhất thì
                # bỏ để giữ tính phản ứng online.
                greedy_dist = INF
                for o in orders.values():
                    if not o.picked and not o.delivered and s.can_carry(o, orders):
                        d = self._dist(s.position, (o.sx, o.sy))
                        if d < greedy_dist:
                            greedy_dist = d

                if vrp_dist <= max(greedy_dist * 1.5, greedy_dist + 3):
                    return self._planned_stop_action(stop, s, obs)

        # --- Priority 2: Deliver đơn đang mang nếu route trống/hỏng ---
        carried = [
            orders[oid]
            for oid in s.bag
            if oid in orders and not orders[oid].delivered
        ]
        if carried:
            best = self._select_delivery_order(s, orders)
            if best is None:
                return ("S", 0)
            goal = (best.ex, best.ey)
            return self._navigate_to(s.position, goal, cargo_op_at_goal=2)

        # --- Priority 3: Greedy fallback ---
        return self._greedy_pick(s, obs)


    def _navigate_to(self, pos: Position, goal: Position, cargo_op_at_goal: int) -> Action:
        """Di chuyển 1 bước về phía goal. Nếu đã tới → thực hiện cargo_op."""
        if pos == goal:
            return ("S", cargo_op_at_goal)
        moves = self._path(pos, goal)
        if not moves:
            return ("S", 0)
        mv = moves[0]
        nxt = valid_next_pos(pos, mv, self.grid)
        if nxt == goal:
            return (mv, cargo_op_at_goal)
        return (mv, 0)

    def _live_cargo_op(self, pos: Position, s: Shipper, obs: dict) -> int:
        """Xác định cargo_op tại pos từ live obs."""
        orders = obs["orders"]
        for oid in s.bag:
            if oid in orders:
                o = orders[oid]
                if not o.delivered and (o.ex, o.ey) == pos:
                    return 2
        for o in orders.values():
            if not o.picked and not o.delivered and (o.sx, o.sy) == pos:
                if s.can_carry(o, orders):
                    return 1
        return 0

    def _greedy_pick(self, s: Shipper, obs: dict) -> Action:
        """Greedy: chọn đơn gần nhất chưa picked (với deadline filter + batch reserve)."""
        orders = obs["orders"]
        obs_t = obs["t"]

        cands = []
        for o in orders.values():
            if o.picked or o.delivered:
                continue
            if o.id in self._reserved:
                continue
            if not s.can_carry(o, orders):
                continue
            d_pickup = self._dist(s.position, (o.sx, o.sy))
            if d_pickup >= INF:
                continue
            # Deadline filter: bỏ đơn chắc chắn reward = 0
            d_deliver = self._dist((o.sx, o.sy), (o.ex, o.ey))
            if obs_t + d_pickup + d_deliver - o.et > self.max_delivery_delay:
                continue
            score = self._pickup_score(s.position, o, obs_t)
            if score <= 0.0:
                continue
            cands.append((score, d_pickup, o))

        if cands:
            _, _, best = max(
                cands,
                key=lambda item: (item[0], -item[1], -item[2].id),
            )
            self._reserved.add(best.id)

            # Batch reserve: reserve thêm đơn "trên đường" để tránh shipper khác cướp
            remaining_slots = s.K_max - len(s.bag) - 1
            w_carried = sum(orders[oid].w for oid in s.bag if oid in orders) + best.w
            for o2 in orders.values():
                if remaining_slots <= 0:
                    break
                if o2.id == best.id or o2.id in self._reserved:
                    continue
                if o2.picked or o2.delivered:
                    continue
                if w_carried + o2.w > s.W_max:
                    continue
                d_direct = self._dist(s.position, (best.sx, best.sy))
                d_via = (self._dist(s.position, (o2.sx, o2.sy))
                         + self._dist((o2.sx, o2.sy), (best.sx, best.sy)))
                if d_via - d_direct > 3:  # BATCH_DETOUR_LIMIT
                    continue
                d_deliver2 = self._dist((o2.sx, o2.sy), (o2.ex, o2.ey))
                if obs_t + d_via + d_deliver2 - o2.et > self.max_delivery_delay:
                    continue
                self._reserved.add(o2.id)
                w_carried += o2.w
                remaining_slots -= 1

            goal = (best.sx, best.sy)
            return self._navigate_to(s.position, goal, cargo_op_at_goal=1)

        return ("S", 0)


    # -----------------------------------------------------------------------
    # Re-planning controller
    # -----------------------------------------------------------------------

    def _should_replan(self, obs: dict) -> bool:
        t = obs["t"]
        if t - self._last_plan_t >= self._replan_interval:
            return True
        if obs.get("new_order_ids"):
            return True
        shippers = obs["shippers"]
        orders = obs["orders"]
        has_unassigned = any(
            not o.picked and not o.delivered for o in orders.values()
        )
        if has_unassigned:
            for s in shippers:
                if not self._targets.get(s.id):
                    return True
        return False

    def _replan(self, obs: dict) -> None:
        self._last_plan_t = obs["t"]
        routes = self._run_vrp(obs)
        if routes is not None:
            new_targets = self._extract_targets(routes, obs)
            for s_id, tgts in new_targets.items():
                self._targets[s_id] = tgts

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        self._targets: Dict[int, deque] = {}
        self._reserved: set = set()
        for s in obs["shippers"]:
            self._targets[s.id] = deque()

        self._last_plan_t = -self._replan_interval

        while not obs.get("done", False):
            if self._should_replan(obs):
                self._replan(obs)

            self._reserved = set()
            actions: Dict[int, Action] = {}
            for s in sorted(obs["shippers"], key=lambda s: s.id):
                actions[s.id] = self._step_action(s, obs)

            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
