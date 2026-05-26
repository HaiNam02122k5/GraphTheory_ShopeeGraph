from __future__ import annotations

import random
import time
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

from env import DeliveryEnv, Order, Shipper, delivery_reward, is_valid_cell, valid_next_pos
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, int]
RouteStop = Tuple[Position, int, int]  # (target position, cargo op, order id)
EdgeKey = Tuple[int, int, int]

INF = 10**9
MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")

REPLAN_INTERVAL_SMALL = 10
REPLAN_INTERVAL_LARGE = 20
MAX_ACTIVE_ORDERS = 45
MAX_STOPS_SMALL = 4
MAX_STOPS_LARGE = 8
MIN_EXPECTED_REWARD = 0.15

ACO_ALPHA = 1.0
ACO_BETA = 2.4
ACO_RHO = 0.35
ACO_Q = 3.0
TAU0 = 1.0
TAU_MIN = 0.05
TAU_MAX = 8.0


from solvers.pathfinder import get_pathfinder


class ACOSolver(Solver):
    """
    Rolling-horizon Ant Colony Optimization cho Online MAPD.

    Mỗi lần replan, solver chạy nhiều colony iterations. Mỗi solution gồm route
    nhiều stop cho toàn bộ shipper; pheromone chỉ sống trong một replan cycle vì
    đơn hàng online thay đổi liên tục.
    """

    method_name = "ACO"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.pathfinder = get_pathfinder(self.grid)
        self._targets: Dict[int, Deque[RouteStop]] = {}
        self._reserved: Set[int] = set()
        self._last_plan_t = -99
        self._replan_interval = (
            REPLAN_INTERVAL_LARGE if env.N >= 18 else REPLAN_INTERVAL_SMALL
        )
        seed = 7919 + env.N * 101 + env.C * 17 + env.G
        self._rng = random.Random(seed)
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

    # ------------------------------------------------------------------
    # BFS helpers (Tăng tốc bởi GPU PathFinder)
    # ------------------------------------------------------------------

    def _bfs(self, start: Position, goal: Position) -> Tuple[int, List[Move]]:
        return self.pathfinder.dist(start, goal), self.pathfinder.path(start, goal)

    def _dist(self, a: Position, b: Position) -> int:
        return self.pathfinder.dist(a, b)

    def _path(self, a: Position, b: Position) -> List[Move]:
        return self.pathfinder.path(a, b)

    # ------------------------------------------------------------------
    # ACO planning
    # ------------------------------------------------------------------

    def _n_iterations(self) -> int:
        if self.env.N <= 10:
            return 14
        if self.env.N <= 15:
            return 10
        return 12

    def _route_limit(self) -> int:
        if self.env.N < 12:
            return 4
        return 8

    def _exploit_probability(self) -> float:
        if self.env.N <= 10:
            return 0.35
        if self.env.N >= 18:
            return 0.55
        if self.env.N == 12:
            return 0.55
        return 0.35

    def _expected_reward(self, from_pos: Position, order: Order, obs_t: int) -> float:
        d_pick = self._dist(from_pos, (order.sx, order.sy))
        d_drop = self._dist((order.sx, order.sy), (order.ex, order.ey))
        if d_pick >= INF or d_drop >= INF:
            return 0.0
        return delivery_reward(order, obs_t + d_pick + d_drop, self.env.T)

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
        reward = delivery_reward(order, eta_delivery, self.env.T)
        if eta_delivery - order.et > self.max_delivery_delay:
            return 0.0
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
        return min(
            carried,
            key=lambda o: (
                0 if obs_t + self._dist(shipper.position, (o.ex, o.ey)) <= o.et else 1,
                o.et if obs_t + self._dist(shipper.position, (o.ex, o.ey)) <= o.et else self._dist(shipper.position, (o.ex, o.ey)),
                self._dist(shipper.position, (o.ex, o.ey)),
                -o.p,
                o.id,
            ),
        )

    def _active_orders(self, obs: dict) -> List[Order]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        obs_t = obs["t"]

        candidates: List[Order] = []
        for order in orders.values():
            if order.picked or order.delivered:
                continue
            nearest = min(
                (s.position for s in shippers),
                key=lambda p: self._dist(p, (order.sx, order.sy)),
            )
            if self._expected_reward(nearest, order, obs_t) < MIN_EXPECTED_REWARD:
                continue
            candidates.append(order)

        def _heuristic_score(o: Order) -> float:
            nearest_p = min(
                (s.position for s in shippers),
                key=lambda p: self._dist(p, (o.sx, o.sy))
            )
            dist = self._dist(nearest_p, (o.sx, o.sy)) + 1
            reward = self._expected_reward(nearest_p, o, obs_t)
            return reward / dist

        candidates.sort(
            key=lambda o: (
                -_heuristic_score(o),
                o.et,
                -o.p,
                o.id,
            )
        )
        max_active = max(45, self.env.C * 8)
        return candidates[:max_active]

    def _node_id(self, stop: RouteStop) -> int:
        _, op, oid = stop
        return oid * 3 + op

    def _edge_key(self, from_node: int, stop: RouteStop) -> EdgeKey:
        return (from_node, self._node_id(stop), stop[1])

    def _eta_pickup(
        self,
        pos: Position,
        order: Order,
        elapsed: int,
        obs_t: int,
        d_pick: int,
    ) -> float:
        d_drop = self._dist((order.sx, order.sy), (order.ex, order.ey))
        if d_pick >= INF or d_drop >= INF:
            return 0.0
        eta_delivery = obs_t + elapsed + d_pick + d_drop
        reward = delivery_reward(order, eta_delivery, self.env.T)
        if reward < MIN_EXPECTED_REWARD:
            return 0.0

        lateness = max(0, eta_delivery - order.et)
        if 12 <= self.env.N < 18 and lateness > self.env.T * 0.25:
            return 0.0

        distance_value = reward / (d_pick + d_drop + 1)
        pickup_bonus = 0.35 * reward / (d_pick + 1)
        deadline_factor = 1.0
        if eta_delivery > order.et:
            deadline_factor -= min(0.5, lateness / max(self.env.T, 1))
        return max(0.001, (distance_value + pickup_bonus) * deadline_factor + 0.05 * order.p)

    def _eta_delivery(
        self,
        order: Order,
        elapsed: int,
        obs_t: int,
        d: int,
    ) -> float:
        if d >= INF:
            return 0.0
        arrival = obs_t + elapsed + d
        reward = delivery_reward(order, arrival, self.env.T)
        urgency = 1.0
        if arrival > order.et:
            urgency += min(1.0, (arrival - order.et) / max(self.env.T, 1))
        return max(0.001, (reward * urgency) / (d + 1))

    def _select_parallel_candidate(
        self,
        candidates: List[Tuple[int, RouteStop, Order, int, float, EdgeKey]],
        pheromone: Dict[EdgeKey, float],
        exploit: bool,
    ) -> int:
        if exploit:
            return max(
                range(len(candidates)),
                key=lambda idx: (
                    pheromone.get(candidates[idx][5], TAU0) ** ACO_ALPHA
                    * candidates[idx][4] ** ACO_BETA,
                    candidates[idx][4],
                    -candidates[idx][3],
                    -candidates[idx][2].id,
                ),
            )

        weights: List[float] = []
        total = 0.0
        for _, _, _, _, eta, edge in candidates:
            weight = pheromone.get(edge, TAU0) ** ACO_ALPHA * eta ** ACO_BETA
            weights.append(weight)
            total += weight

        if total <= 0:
            return self._rng.choice(range(len(candidates)))

        pick = self._rng.random() * total
        acc = 0.0
        for idx, weight in enumerate(weights):
            acc += weight
            if acc >= pick:
                return idx
        return len(candidates) - 1

    def _construct_solution(
        self,
        obs: dict,
        active_orders: List[Order],
        pheromone: Dict[EdgeKey, float],
        exploit: bool,
    ) -> Dict[int, List[RouteStop]]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        obs_t = obs["t"]

        assigned: Set[int] = set()
        routes: Dict[int, List[RouteStop]] = {s.id: [] for s in shippers}

        state: Dict[int, dict] = {}
        for s in shippers:
            bag = [oid for oid in s.bag if oid in orders and not orders[oid].delivered]
            state[s.id] = {
                "pos": s.position,
                "elapsed": 0,
                "from_node": -(s.id + 1),
                "bag": bag,
                "carried_weight": sum(orders[oid].w for oid in bag if oid in orders),
                "finished": False,
                "K_max": s.K_max,
                "W_max": s.W_max,
            }

        while True:
            candidates: List[Tuple[int, RouteStop, Order, int, float, EdgeKey]] = []

            for sid, s_state in state.items():
                if s_state["finished"] or len(routes[sid]) >= self._route_limit():
                    continue

                pos = s_state["pos"]
                elapsed = s_state["elapsed"]
                from_node = s_state["from_node"]
                bag = s_state["bag"]
                carried_weight = s_state["carried_weight"]
                K_max = s_state["K_max"]
                W_max = s_state["W_max"]

                for oid in list(bag):
                    order = orders.get(oid)
                    if order is None or order.delivered:
                        continue
                    stop: RouteStop = ((order.ex, order.ey), 2, order.id)
                    d = self._dist(pos, stop[0])
                    eta = self._eta_delivery(order, elapsed, obs_t, d)
                    if eta > 0:
                        candidates.append((sid, stop, order, d, eta, self._edge_key(from_node, stop)))

                if len(bag) < K_max:
                    for order in active_orders:
                        if order.id in assigned or order.id in bag:
                            continue
                        if order.picked or order.delivered:
                            continue
                        if carried_weight + order.w > W_max:
                            continue
                        stop = ((order.sx, order.sy), 1, order.id)
                        d_pick = self._dist(pos, stop[0])
                        eta = self._eta_pickup(pos, order, elapsed, obs_t, d_pick)
                        if eta > 0:
                            candidates.append((sid, stop, order, d_pick, eta, self._edge_key(from_node, stop)))

            if not candidates:
                break

            selected_idx = self._select_parallel_candidate(candidates, pheromone, exploit)
            sid, stop, order, travel, _, edge_key = candidates[selected_idx]

            s_state = state[sid]
            routes[sid].append(stop)
            s_state["pos"] = stop[0]
            s_state["elapsed"] += travel
            s_state["from_node"] = self._node_id(stop)

            if stop[1] == 1:
                assigned.add(order.id)
                s_state["bag"].append(order.id)
                s_state["carried_weight"] += order.w
            else:
                if order.id in s_state["bag"]:
                    s_state["bag"].remove(order.id)
                s_state["carried_weight"] = max(0.0, s_state["carried_weight"] - order.w)

            if s_state["elapsed"] >= self.env.T:
                s_state["finished"] = True

        return routes

    def _score_routes(self, routes: Dict[int, List[RouteStop]], obs: dict) -> float:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        obs_t = obs["t"]
        shipper_map = {s.id: s for s in shippers}

        score = 0.0
        delivered: Set[int] = set()
        picked: Set[int] = set()

        for sid, route in routes.items():
            shipper = shipper_map[sid]
            pos = shipper.position
            elapsed = 0
            bag = [oid for oid in shipper.bag if oid in orders and not orders[oid].delivered]
            carried_weight = sum(orders[oid].w for oid in bag if oid in orders)

            for stop_pos, op, oid in route:
                order = orders.get(oid)
                if order is None or order.delivered:
                    continue
                d = self._dist(pos, stop_pos)
                if d >= INF:
                    score -= 100.0
                    continue

                move_penalty = 0.01 * d * (1.0 + carried_weight / max(shipper.W_max, 1.0))
                elapsed += d
                pos = stop_pos
                score -= move_penalty

                if op == 1:
                    if oid in picked or oid in bag:
                        score -= 5.0
                        continue
                    if len(bag) >= shipper.K_max or carried_weight + order.w > shipper.W_max:
                        score -= 20.0
                        continue
                    picked.add(oid)
                    bag.append(oid)
                    carried_weight += order.w
                elif op == 2:
                    if oid not in bag:
                        score -= 10.0
                        continue
                    reward = delivery_reward(order, obs_t + elapsed, self.env.T)
                    score += reward
                    delivered.add(oid)
                    bag.remove(oid)
                    carried_weight = max(0.0, carried_weight - order.w)

        return score + 0.01 * len(delivered)

    def _update_pheromone(
        self,
        pheromone: Dict[EdgeKey, float],
        routes: Dict[int, List[RouteStop]],
        score: float,
    ) -> None:
        for edge in list(pheromone):
            pheromone[edge] = max(TAU_MIN, pheromone[edge] * (1.0 - ACO_RHO))

        deposit = max(0.01, ACO_Q * max(score, 1.0) / 100.0)
        for sid, route in routes.items():
            from_node = -(sid + 1)
            for stop in route:
                edge = self._edge_key(from_node, stop)
                pheromone[edge] = min(TAU_MAX, pheromone.get(edge, TAU0) + deposit)
                from_node = self._node_id(stop)

    def _run_aco(self, obs: dict) -> Optional[Dict[int, List[RouteStop]]]:
        active_orders = self._active_orders(obs)
        has_carried = any(s.bag for s in obs["shippers"])
        if not active_orders and not has_carried:
            return None

        pheromone: Dict[EdgeKey, float] = {}
        best_routes: Optional[Dict[int, List[RouteStop]]] = None
        best_score = -float("inf")

        # Iteration 0 is deterministic greedy construction. Later iterations
        # explore around it with pheromone-guided sampling.
        for iteration in range(self._n_iterations()):
            exploit = iteration == 0 or self._rng.random() < self._exploit_probability()
            routes = self._construct_solution(obs, active_orders, pheromone, exploit=exploit)
            score = self._score_routes(routes, obs)

            if score > best_score:
                best_score = score
                best_routes = routes
            if best_routes is not None:
                self._update_pheromone(pheromone, best_routes, best_score)

        return best_routes

    # ------------------------------------------------------------------
    # Route execution
    # ------------------------------------------------------------------

    def _is_stop_actionable(self, stop: RouteStop, shipper: Shipper, obs: dict) -> bool:
        pos, op, oid = stop
        orders: Dict[int, Order] = obs["orders"]
        order = orders.get(oid)
        if order is None or order.delivered:
            return False
        if op == 1:
            return (
                not order.picked
                and (order.sx, order.sy) == pos
                and shipper.can_carry(order, orders)
            )
        if op == 2:
            return (
                oid in shipper.bag
                and not order.delivered
                and (order.ex, order.ey) == pos
            )
        return False

    def _navigate_to(self, pos: Position, goal: Position, cargo_op_at_goal: int) -> Action:
        if pos == goal:
            return ("S", cargo_op_at_goal)
        moves = self._path(pos, goal)
        if not moves:
            return ("S", 0)
        move = moves[0]
        nxt = valid_next_pos(pos, move, self.grid)
        if nxt == goal:
            return (move, cargo_op_at_goal)
        return (move, 0)

    def _greedy_pick(self, shipper: Shipper, obs: dict) -> Action:
        orders: Dict[int, Order] = obs["orders"]
        obs_t = obs["t"]
        candidates: List[Tuple[float, int, Order]] = []

        for order in orders.values():
            if order.picked or order.delivered:
                continue
            if order.id in self._reserved:
                continue
            if not shipper.can_carry(order, orders):
                continue
            d_pick = self._dist(shipper.position, (order.sx, order.sy))
            d_drop = self._dist((order.sx, order.sy), (order.ex, order.ey))
            if d_pick >= INF or d_drop >= INF:
                continue
            if obs_t + d_pick + d_drop - order.et > self.max_delivery_delay:
                continue
            score = self._pickup_score(shipper.position, order, obs_t)
            if score <= 0.0:
                continue
            candidates.append((score, d_pick, order))

        if not candidates:
            return ("S", 0)

        _, _, best = max(
            candidates,
            key=lambda item: (item[0], -item[1], -item[2].id),
        )
        self._reserved.add(best.id)

        remaining_slots = shipper.K_max - len(shipper.bag) - 1
        w_carried = sum(orders[oid].w for oid in shipper.bag if oid in orders) + best.w
        d_direct = self._dist(shipper.position, (best.sx, best.sy))
        for other in orders.values():
            if remaining_slots <= 0:
                break
            if other.id == best.id or other.id in self._reserved:
                continue
            if other.picked or other.delivered:
                continue
            if w_carried + other.w > shipper.W_max:
                continue
            d_via = (
                self._dist(shipper.position, (other.sx, other.sy))
                + self._dist((other.sx, other.sy), (best.sx, best.sy))
            )
            if d_via - d_direct > 3:
                continue
            d_drop = self._dist((other.sx, other.sy), (other.ex, other.ey))
            if obs_t + d_via + d_drop - other.et > self.max_delivery_delay:
                continue
            self._reserved.add(other.id)
            w_carried += other.w
            remaining_slots -= 1

        return self._navigate_to(shipper.position, (best.sx, best.sy), 1)

    def _step_action(self, shipper: Shipper, obs: dict) -> Action:
        orders: Dict[int, Order] = obs["orders"]
        queue = self._targets.get(shipper.id)

        if queue:
            while queue and not self._is_stop_actionable(queue[0], shipper, obs):
                queue.popleft()

            if queue:
                stop = queue[0]
                goal, op, _ = stop

                if op == 2:
                    return self._navigate_to(shipper.position, goal, op)

                greedy_dist = INF
                for order in orders.values():
                    if not order.picked and not order.delivered and shipper.can_carry(order, orders):
                        greedy_dist = min(greedy_dist, self._dist(shipper.position, (order.sx, order.sy)))
                
                bypass_factor = 2.2 if self.env.N >= 25 else 3.5
                bypass_offset = 8 if self.env.N >= 25 else 15
                if self._dist(shipper.position, goal) <= max(greedy_dist * bypass_factor, greedy_dist + bypass_offset):
                    return self._navigate_to(shipper.position, goal, op)

        carried = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if carried:
            best = self._select_delivery_order(shipper, orders)
            if best is None:
                return ("S", 0)
            return self._navigate_to(shipper.position, (best.ex, best.ey), 2)

        return self._greedy_pick(shipper, obs)

    def _should_replan(self, obs: dict) -> bool:
        t = obs["t"]
        if t - self._last_plan_t >= self._replan_interval:
            return True
        if obs.get("new_order_ids"):
            return True
        if any(not o.picked and not o.delivered for o in obs["orders"].values()):
            return any(not self._targets.get(s.id) for s in obs["shippers"])
        return False

    def _replan(self, obs: dict) -> None:
        self._last_plan_t = obs["t"]
        routes = self._run_aco(obs)
        if routes is None:
            return
        for sid, route in routes.items():
            self._targets[sid] = deque(route)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        self._targets = {s.id: deque() for s in obs["shippers"]}
        self._reserved = set()
        self._last_plan_t = -self._replan_interval

        while not obs.get("done", False):
            if self._should_replan(obs):
                self._replan(obs)

            self._reserved = set()
            actions: Dict[int, Action] = {}
            for shipper in sorted(obs["shippers"], key=lambda s: s.id):
                actions[shipper.id] = self._step_action(shipper, obs)

            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
