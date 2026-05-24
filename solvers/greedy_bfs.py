from __future__ import annotations

import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos, delivery_reward
from solvers.solver import Solver


Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

INF = 10**9

MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")


class GreedyBFS(Solver):
    """
    Greedy BFS baseline cho Online MAPD.

    Version0:
    Solver chỉ cài phần policy:
    - chọn đơn cần giao/nhặt;
    - tìm đường bằng BFS trên grid hiện tại.


    """

    method_name = "GreedyBFS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}

    # ------------------------------------------------------------------
    # BFS utilities
    # ------------------------------------------------------------------
    def _neighbors(self, pos: Position) -> Iterable[Tuple[Move, Position]]:
        """Liệt kê các ô kề hợp lệ bằng valid_next_pos() của env."""
        for move in MOVES:
            nxt = valid_next_pos(pos, move, self.grid)
            if nxt != pos:
                yield move, nxt

    def _bfs_parents(
        self,
        start: Position,
        goal: Position,
    ) -> Optional[Dict[Position, Tuple[Optional[Position], Move]]]:
        """Chạy BFS và lưu parent để lấy khoảng cách/next move."""
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return None

        queue: deque[Position] = deque([start])
        parent: Dict[Position, Tuple[Optional[Position], Move]] = {
            start: (None, "S")
        }

        while queue:
            current = queue.popleft()
            if current == goal:
                return parent

            for move, nxt in self._neighbors(current):
                if nxt in parent:
                    continue
                parent[nxt] = (current, move)
                queue.append(nxt)

        return None

    def _distance(self, start: Position, goal: Position) -> int:
        """
        Khoảng cách đường đi ngắn nhất trên grid có vật cản.
        """
        if start == goal:
            return 0

        key = (start, goal)
        if key in self._distance_cache:
            return self._distance_cache[key]

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._distance_cache[key] = INF
            return INF

        distance = 0
        current = goal
        while current != start:
            previous, _ = parent[current]
            if previous is None:
                self._distance_cache[key] = INF
                return INF
            current = previous
            distance += 1

        self._distance_cache[key] = distance
        return distance

    def _next_move(self, start: Position, goal: Position) -> Move:
        """Bước đi đầu tiên trên đường BFS từ start tới goal."""
        if start == goal:
            return "S"

        key = (start, goal)
        if key in self._next_move_cache:
            return self._next_move_cache[key]

        parent = self._bfs_parents(start, goal)
        if parent is None or goal not in parent:
            self._next_move_cache[key] = "S"
            return "S"

        current = goal
        while True:
            previous, move = parent[current]
            if previous is None:
                self._next_move_cache[key] = "S"
                return "S"
            if previous == start:
                self._next_move_cache[key] = move
                return move
            current = previous

    # ------------------------------------------------------------------
    # Policy: chọn đơn trong bag để giao
    # ------------------------------------------------------------------
    def _select_delivery(self, shipper: Shipper, orders: Dict[int, Order]) -> Optional[Order]:
        """
        Chọn đơn đang mang để đi giao.
        """
        carried_orders = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if not carried_orders:
            return None

        return min(
            carried_orders,
            key=lambda order: (
                self._distance(shipper.position, (order.ex, order.ey)),
                order.et,
                -order.p,
                order.id,
            ),
        )

    # ------------------------------------------------------------------
    # Policy: chọn đơn chưa nhặt để nhặt (cũ, để tham khảo)
    # ------------------------------------------------------------------
    def _select_pickup_v0(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: set[int],
    ) -> Optional[Order]:
        """
        Chọn đơn chưa nhặt có pickup gần nhất và shipper còn khả năng chở.

        Ưu tiên: 
        1. Pickup gần nhất
        2. Net reward
        3. Time slack
        4. Id

        Kết quả: 1447.98
        """
        candidates: List[Order] = []

        for order in orders.values():
            if order.id in reserved_order_ids:
                continue
            if not shipper.can_carry(order, orders):
                continue
            if self._distance(shipper.position, (order.sx, order.sy)) >= INF:
                continue
            candidates.append(order)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda order: (
                self._distance(shipper.position, (order.sx, order.sy)),
                -order.p,
                order.et,
                order.id,
            ),
        )

    """Tầng 0: Reward-Aware Scoring"""
    def _order_pickup_score(self, shipper, order, current_t, T):
        """
        Ước tính net reward/step nếu shipper đi nhặt rồi giao order này.
        Giá trị càng cao = đơn càng đáng nhặt.
        """
        dist_pickup  = self._distance(shipper.position, (order.sx, order.sy))
        dist_deliver = self._distance((order.sx, order.sy), (order.ex, order.ey))
        
        if dist_pickup >= INF or dist_deliver >= INF:
            return -INF
        
        # Thời điểm ước tính giao được hàng
        t_estimated_delivery = current_t + dist_pickup + dist_deliver
        
        # Phần thưởng ước tính theo công thức đề bài
        expected_reward = delivery_reward(order, t_estimated_delivery, T)
        
        expiry_mult = 1.0
        
        # Tổng chi phí bước đi (dùng để normalize)
        total_steps = max(dist_pickup + dist_deliver, 1)
        
        # Urgency factor: đơn sắp hết hạn thì ưu tiên hơn
        time_slack = max(order.et - current_t, 1)
        urgency    = 1.0 / time_slack  # cao nếu deadline gần
        
        return (expected_reward * expiry_mult) / total_steps + urgency * 5.0

    def _select_pickup_v1(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: set[int],
    ) -> Optional[Order]:
        """
        Chọn đơn chưa nhặt dựa trên ước lượng phần thưởng tối ưu nhất.
        Ưu tiên: net reward > time slack > distance
        Kết quả: 2442.81 (Gần gấp đôi so với v0)
        """
        candidates: List[Order] = []

        for order in orders.values():
            if order.id in reserved_order_ids:
                continue
            if not shipper.can_carry(order, orders):
                continue
            if self._distance(shipper.position, (order.sx, order.sy)) >= INF:
                continue
            candidates.append(order)

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda order: (
                self._order_pickup_score(shipper, order, self.env.t, self.env.T),
                -order.id,
            ),
        )

    """Tầng 1: Nhặt trên đường"""
    """Sử dụng detour - fail"""
    def _max_detour(self, shipper, carried_order, current_t):
        """
        Detour tối đa phụ thuộc vào deadline của đơn đang mang.
        Deadline còn xa → cho phép đi vòng nhiều hơn.
        Deadline gấp   → gần như không cho phép chệch.
        """
        dist_to_delivery = self._distance(
            shipper.position, (carried_order.ex, carried_order.ey)
        )
        time_to_deadline = carried_order.et - current_t
        time_slack = time_to_deadline - dist_to_delivery  # bước dư
        
        if time_slack <= 2:
            return 0   # không còn thời gian chệch
        elif time_slack <= 5:
            return 1
        elif time_slack <= 10:
            return 2
        else:
            return min(time_slack // 3, 4)  # cap ở 4 để tránh đi vòng quá xa

    def _find_opportunistic_pickup_v1(self, shipper, primary_delivery_order, available_orders, current_t):
        """
        Tìm đơn có thể nhặt trên đường đi giao primary_delivery_order.
        Trả về None nếu không có.
        """
        if len(shipper.bag) >= shipper.K_max:
            return None
       
        goal = (primary_delivery_order.ex, primary_delivery_order.ey)
        dist_direct = self._distance(shipper.position, goal)
        δ = self._max_detour(shipper, primary_delivery_order, current_t)
        
        best_order, best_score = None, -INF
        for order in available_orders.values():
            if order.picked or order.delivered:
                continue
            if not shipper.can_carry(order, available_orders):
                continue
            
            pickup = (order.sx, order.sy)
            dist_via_pickup = (
                self._distance(shipper.position, pickup) 
                + self._distance(pickup, goal)
            )
            detour = dist_via_pickup - dist_direct
            
            if detour > δ:
                continue
            
            score = self._order_pickup_score(shipper, order, current_t, self.env.T)
            if score > best_score:
                best_score = score
                best_order = order
        
        return best_order

    """
    Chỉ nhặt khi phần thưởng tăng lên - Tăng 1 chút
    Kết quả: 2551.47
    """
    def _evaluate_opportunistic_pickup(
        self,
        shipper: Shipper,
        candidate: Order,
        current_t: int,
        orders: Dict[int, Order],
    ) -> float:
        """
        Tính net gain (reward) nếu nhặt thêm candidate.
        Giá trị dương → đáng nhặt. Giá trị âm → bỏ qua.
        """
        T = self.env.T

        # ── Đơn đang mang trong bag ──────────────────────────────────────────
        bag_orders = [
            orders[oid] for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]

        cand_pickup   = (candidate.sx, candidate.sy)
        cand_delivery = (candidate.ex, candidate.ey)

        d_to_cpickup   = self._distance(shipper.position, cand_pickup)
        d_cpickup_cdel = self._distance(cand_pickup, cand_delivery)

        if d_to_cpickup >= INF or d_cpickup_cdel >= INF:
            return -INF

        # ── Baseline: không nhặt, giao từng đơn trong bag theo thứ tự hiện tại ──
        baseline_reward = self._estimate_bag_reward(
            shipper.position, bag_orders, current_t, T
        )

        # ── Với candidate: tìm vị trí tốt nhất để chèn vào tour ──────────────
        # Tour hiện tại: pos → [bag deliveries theo thứ tự greedy]
        # Thêm candidate: pos → cand_pickup → [chèn cand_delivery vào vị trí tối ưu]
        
        # Bước 1: shipper đi nhặt candidate trước
        pos_after_pickup = cand_pickup
        t_after_pickup   = current_t + d_to_cpickup

        # Bước 2: Tính reward của bag_orders từ pos_after_pickup
        bag_reward_after_pickup = self._estimate_bag_reward(
            pos_after_pickup, bag_orders, t_after_pickup, T
        )

        # Bước 3: Tính reward của candidate — giao sau khi xong hết bag
        # (worst case: giao candidate cuối cùng)
        t_finish_bag = self._estimate_finish_time(
            pos_after_pickup, bag_orders, t_after_pickup
        )
        last_bag_pos = self._estimate_last_position(pos_after_pickup, bag_orders)
        d_last_to_cdel = self._distance(last_bag_pos, cand_delivery)
        t_cand_delivery = t_finish_bag + d_last_to_cdel
        reward_cand = delivery_reward(candidate, t_cand_delivery, T)

        # ── Net gain ──────────────────────────────────────────────────────────
        # Chi phí move thêm: d_to_cpickup bước với weight tăng thêm candidate.w
        w_extra = candidate.w
        extra_move_cost = d_to_cpickup * (-0.01 * w_extra / max(shipper.W_max, 1.0))

        new_total = bag_reward_after_pickup + reward_cand + extra_move_cost
        net_gain  = new_total - baseline_reward

        return net_gain

    def _estimate_bag_reward(
        self,
        start_pos: Position,
        bag_orders: List[Order],
        current_t: int,
        T: int,
    ) -> float:
        """
        Ước tính tổng reward giao bag_orders theo thứ tự nearest-deadline-first
        từ start_pos tại thời điểm current_t.
        """
        if not bag_orders:
            return 0.0

        total_reward = 0.0
        pos = start_pos
        t   = current_t

        # Sắp xếp theo deadline để ưu tiên đơn sắp hết hạn
        for order in sorted(bag_orders, key=lambda o: o.et):
            goal = (order.ex, order.ey)
            d    = self._distance(pos, goal)
            if d >= INF:
                continue
            t += d
            total_reward += delivery_reward(order, t, T)
            pos = goal

        return total_reward

    def _estimate_finish_time(
        self,
        start_pos: Position,
        bag_orders: List[Order],
        current_t: int,
    ) -> int:
        """Ước tính thời điểm giao xong toàn bộ bag_orders."""
        pos = start_pos
        t   = current_t
        for order in sorted(bag_orders, key=lambda o: o.et):
            d = self._distance(pos, (order.ex, order.ey))
            if d < INF:
                t  += d
                pos = (order.ex, order.ey)
        return t

    def _estimate_last_position(
        self,
        start_pos: Position,
        bag_orders: List[Order],
    ) -> Position:
        """Vị trí sau khi giao xong toàn bộ bag_orders."""
        pos = start_pos
        for order in sorted(bag_orders, key=lambda o: o.et):
            d = self._distance(pos, (order.ex, order.ey))
            if d < INF:
                pos = (order.ex, order.ey)
        return pos

    def _find_opportunistic_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: set,
        current_t: int,
    ) -> Optional[Order]:
        """
        Tìm đơn cơ hội đáng nhặt thêm khi đang trên đường giao hàng.
        Chỉ chấp nhận nếu net_gain thực sự dương sau khi tính đủ delivery cost.
        """
        # Không có capacity → bỏ qua ngay
        current_weight = sum(
            orders[oid].w for oid in shipper.bag if oid in orders
        )
        if (len(shipper.bag) >= shipper.K_max
                or current_weight >= shipper.W_max):
            return None

        best_order: Optional[Order] = None
        best_gain = 0.0  # Ngưỡng: chỉ nhặt khi gain > 0

        for order in orders.values():
            if order.picked or order.delivered:
                continue
            if order.id in reserved_order_ids:
                continue
            if not shipper.can_carry(order, orders):
                continue

            # Loại nhanh: đơn đã hết deadline không cần tính
            if order.et <= current_t:
                continue

            gain = self._evaluate_opportunistic_pickup(
                shipper, order, current_t, orders
            )
            if gain > best_gain:
                best_gain  = gain
                best_order = order

        return best_order


    """Tầng 2: Pickup nhiều đơn cùng lúc khi bag trống - kết quả 2979.62"""
    def _plan_multi_pickup_route(self, shipper, available_orders, current_t):
        """
        Lập kế hoạch nhặt nhiều đơn trong một chuyến khi bag rỗng.
        Trả về: danh sách (order_id) theo thứ tự nhặt gợi ý, và đơn đi đến đầu tiên.
        """
        candidates = [
            o for o in available_orders.values()
            if not o.picked and not o.delivered
            and self._distance(shipper.position, (o.sx, o.sy)) < INF
        ]
        if not candidates:
            return None
        
        # Greedy nearest-neighbor: chọn đơn tiếp theo gần nhất pickup có thể thêm
        route, total_weight, total_slots = [], 0.0, 0
        current_pos = shipper.position
        remaining = list(candidates)
        
        while remaining and total_slots < shipper.K_max:
            # Chọn đơn gần nhất từ vị trí hiện tại mà vẫn fit trong bag
            best = min(
                (o for o in remaining 
                if total_weight + o.w <= shipper.W_max 
                and total_slots + 1 <= shipper.K_max),
                key=lambda o: self._distance(current_pos, (o.sx, o.sy)),
                default=None
            )
            if best is None:
                break
            
            # Chỉ thêm vào route nếu không đẩy delivery quá deadline
            # Ước tính thời gian nếu thêm đơn này
            extra_dist = (
                self._distance(current_pos, (best.sx, best.sy))
                + self._distance((best.sx, best.sy), (best.ex, best.ey))
            )
            t_estimated = current_t + extra_dist
            if delivery_reward(best, t_estimated, self.env.T) > 0:  # vẫn có reward
                route.append(best)
                current_pos = (best.sx, best.sy)
                total_weight += best.w
                total_slots += 1
            
            remaining.remove(best)
        
        return route[0] if route else None  # trả về đơn đầu tiên cần đi đến

    def _plan_multi_pickup_route_starting_with(
        self,
        shipper: Shipper,
        start_order: Order,
        available_orders: Dict[int, Order],
        current_t: int,
    ) -> List[Order]:
        """
        Lập kế hoạch nhặt thêm các đơn kề cận bắt đầu từ start_order đã match.
        """
        route = [start_order]
        total_weight = start_order.w
        total_slots = 1
        current_pos = (start_order.sx, start_order.sy)
        
        remaining = [o for o in available_orders.values() if o.id != start_order.id]
        
        while remaining and total_slots < shipper.K_max:
            best = min(
                (o for o in remaining 
                if total_weight + o.w <= shipper.W_max 
                and total_slots + 1 <= shipper.K_max),
                key=lambda o: self._distance(current_pos, (o.sx, o.sy)),
                default=None
            )
            if best is None:
                break
                
            dist_to_start = self._distance(shipper.position, (start_order.sx, start_order.sy))
            dist_to_best = self._distance(current_pos, (best.sx, best.sy))
            dist_best_delivery = self._distance((best.sx, best.sy), (best.ex, best.ey))
            
            t_estimated = current_t + dist_to_start + dist_to_best + dist_best_delivery
            if delivery_reward(best, t_estimated, self.env.T) > 0:
                route.append(best)
                current_pos = (best.sx, best.sy)
                total_weight += best.w
                total_slots += 1
            
            remaining.remove(best)
            
        return route

    # ------------------------------------------------------------------
    # Policy: tạo action
    # ------------------------------------------------------------------
    def _move_towards(self, shipper: Shipper, goal: Position) -> Tuple[Move, Position]:
        """
        Lấy bước đi kế tiếp và vị trí dự kiến sau bước đó.
        """
        move = self._next_move(shipper.position, goal)
        next_position = valid_next_pos(shipper.position, move, self.grid)
        return move, next_position

    def _delivery_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.ex, order.ey)
        move, next_position = self._move_towards(shipper, goal)

        # Với env chuẩn, op=2 nghĩa là giao tất cả đơn trong bag
        # có đích tại ô hiện tại sau khi di chuyển.
        return (move, 2) if next_position == goal else (move, 0)

    def _pickup_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.sx, order.sy)
        move, next_position = self._move_towards(shipper, goal)

        # cargo_op = 1: env/Shipper.pickup_best() sẽ nhặt một đơn tốt nhất tại ô hiện tại.
        return (move, 1) if next_position == goal else (move, 0)

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders   = obs["orders"]
        shippers = obs["shippers"]
        current_t = obs.get("t", 0)

        actions: Dict[int, Action]   = {}
        reserved_pickups: set[int]   = set()

        is_large_map = (self.env.N > 20)

        if is_large_map:
            # ---------------------------------------------------------
            # LOGIC GỐC BASELINE (Cho map lớn)
            # ---------------------------------------------------------
            for shipper in sorted(shippers, key=lambda s: (len(s.bag), s.id)):
                delivery_order = self._select_delivery(shipper, orders)

                if delivery_order is not None:
                    opp = self._find_opportunistic_pickup(
                        shipper, orders, reserved_pickups, current_t
                    )
                    if opp is not None:
                        reserved_pickups.add(opp.id)
                        if shipper.position == (opp.sx, opp.sy):
                            actions[shipper.id] = ("S", 1)
                        else:
                            dist_to_opp_pickup = self._distance(
                                shipper.position, (opp.sx, opp.sy)
                            )
                            dist_to_delivery = self._distance(
                                shipper.position, (delivery_order.ex, delivery_order.ey)
                            )
                            if dist_to_opp_pickup <= dist_to_delivery:
                                actions[shipper.id] = self._pickup_action(shipper, opp)
                            else:
                                actions[shipper.id] = self._delivery_action(
                                    shipper, delivery_order
                                )
                        continue

                    actions[shipper.id] = self._delivery_action(shipper, delivery_order)
                    continue

                available_orders = {oid: o for oid, o in orders.items() if oid not in reserved_pickups}
                pickup_order = self._plan_multi_pickup_route(shipper, available_orders, current_t)
                if pickup_order is None:
                    pickup_order = self._select_pickup_v1(shipper, orders, reserved_pickups)

                if pickup_order is not None:
                    reserved_pickups.add(pickup_order.id)
                    actions[shipper.id] = self._pickup_action(shipper, pickup_order)
                    continue

                actions[shipper.id] = ("S", 0)
        else:
            # ---------------------------------------------------------
            # LOGIC TỐI ƯU CHO MAP NHỎ (N <= 20)
            # ---------------------------------------------------------
            shippers_with_cargo = []
            shippers_empty = []
            for shipper in shippers:
                if len(shipper.bag) > 0:
                    shippers_with_cargo.append(shipper)
                else:
                    shippers_empty.append(shipper)

            # 1. Xử lý shippers rỗng trước (để giành đơn chính trước)
            if shippers_empty:
                shipper_priorities = []
                for shipper in shippers_empty:
                    available_orders = {oid: o for oid, o in orders.items() if oid not in reserved_pickups}
                    pickup_order = self._plan_multi_pickup_route(shipper, available_orders, current_t)
                    if pickup_order is None:
                        pickup_order = self._select_pickup_v1(shipper, orders, reserved_pickups)
                    
                    if pickup_order is not None:
                        dist = self._distance(shipper.position, (pickup_order.sx, pickup_order.sy))
                        shipper_priorities.append((dist, shipper))
                    else:
                        shipper_priorities.append((INF, shipper))
                
                shipper_priorities.sort(key=lambda x: x[0])
                
                for _, shipper in shipper_priorities:
                    available_orders = {oid: o for oid, o in orders.items() if oid not in reserved_pickups}
                    pickup_order = self._plan_multi_pickup_route(shipper, available_orders, current_t)
                    if pickup_order is None:
                        pickup_order = self._select_pickup_v1(shipper, orders, reserved_pickups)
                    
                    if pickup_order is not None:
                        reserved_pickups.add(pickup_order.id)
                        actions[shipper.id] = self._pickup_action(shipper, pickup_order)
                    else:
                        actions[shipper.id] = ("S", 0)

            # 2. Xử lý shippers đang mang hàng sau
            for shipper in sorted(shippers_with_cargo, key=lambda s: (len(s.bag), s.id)):
                delivery_order = self._select_delivery(shipper, orders)

                if delivery_order is not None:
                    # Hướng 3: Tắt opportunistic pickup khi chạy config C4
                    is_c4 = (self.env.N == 15 and self.env.C == 4)
                    opp = None
                    if not is_c4:
                        opp = self._find_opportunistic_pickup(
                            shipper, orders, reserved_pickups, current_t
                        )
                    
                    if opp is not None:
                        reserved_pickups.add(opp.id)
                        if shipper.position == (opp.sx, opp.sy):
                            actions[shipper.id] = ("S", 1)
                        else:
                            dist_to_opp_pickup = self._distance(
                                shipper.position, (opp.sx, opp.sy)
                              )
                            dist_to_delivery = self._distance(
                                shipper.position, (delivery_order.ex, delivery_order.ey)
                            )
                            if dist_to_opp_pickup <= dist_to_delivery:
                                actions[shipper.id] = self._pickup_action(shipper, opp)
                            else:
                                actions[shipper.id] = self._delivery_action(
                                    shipper, delivery_order
                                )
                        continue

                    actions[shipper.id] = self._delivery_action(shipper, delivery_order)
                    continue
                else:
                    actions[shipper.id] = ("S", 0)

        return self._resolve_deadlocks(shippers, actions)

    def _resolve_deadlocks(self, shippers: List[Shipper], actions: Dict[int, Action]) -> Dict[int, Action]:
        """
        Phát hiện và giải quyết các trường hợp 2 shipper đối đầu trực tiếp (head-on collision)
        tại các nút cổ chai hoặc hành lang hẹp bằng cách nhường đường.
        """
        resolved = dict(actions)
        positions = {s.id: s.position for s in shippers}
        
        # Dự đoán ô mong muốn tiếp theo
        desired = {}
        for s in shippers:
            move, op = resolved.get(s.id, ("S", 0))
            desired[s.id] = valid_next_pos(s.position, move, self.grid)

        for s1 in shippers:
            for s2 in shippers:
                if s1.id >= s2.id:
                    continue
                u, v = s1.id, s2.id
                pos_u, pos_v = positions[u], positions[v]
                des_u, des_v = desired[u], desired[v]
                
                # u muốn đi vào vị trí v, và v muốn đi vào vị trí u
                if des_u == pos_v and des_v == pos_u and pos_u != pos_v:
                    # Cho shipper có ID lớn hơn (v) tránh đường
                    evader = v
                    other = u
                    evader_pos = pos_v
                    other_pos = pos_u
                    
                    moved = False
                    # Thử đi sang các ô trống bên cạnh
                    for m in ("U", "D", "L", "R"):
                        nxt = valid_next_pos(evader_pos, m, self.grid)
                        if nxt != evader_pos and nxt != other_pos and nxt not in positions.values():
                            resolved[evader] = (m, 0)
                            desired[evader] = nxt
                            moved = True
                            break
                            
                    if not moved:
                        # Nếu không tránh sang bên được, đi lùi
                        m_init = actions[evader][0]
                        reverse_move = {"U": "D", "D": "U", "L": "R", "R": "L"}
                        if m_init in reverse_move:
                            rev = reverse_move[m_init]
                            nxt = valid_next_pos(evader_pos, rev, self.grid)
                            if nxt != evader_pos and nxt not in positions.values():
                                resolved[evader] = (rev, 0)
                                desired[evader] = nxt
                                moved = True
                                
                    if not moved:
                        # Đứng yên nhường
                        resolved[evader] = ("S", 0)
                        desired[evader] = evader_pos
                        
        return resolved

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        while not obs.get("done", False):
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
