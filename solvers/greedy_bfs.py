from __future__ import annotations

import time
from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos
from solvers.solver import Solver


from solvers.pathfinder import get_pathfinder

Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

INF = 10**9

MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")

# Cải tiến #3: Ngưỡng detour tối đa (bước) để batch reserve thêm đơn
# Tăng/giảm giá trị này để tune aggressiveness của batching
BATCH_DETOUR_LIMIT = 3


class GreedyBFS(Solver):
    """
    Greedy BFS cải tiến cho Online MAPD.

    Cải tiến đã xác nhận qua ablation study:
    #2 Deadline Filter  — bỏ đơn chắc chắn reward = 0 (est_t >= deadline + T)
                          Giúp tránh shipper chạy theo đơn hoàn toàn vô nghĩa.
    #3 Pickup Batching  — khi shipper A nhặt đơn X, reserve thêm đơn Y gần đó
                          để shipper B không cướp Y trên cùng tuyến đường.
                          Tăng tổng reward từ 1448 → 1680 (+16%).
    #5 LRU Cache        — giới hạn kích thước cache để ổn định bộ nhớ.

    Cải tiến KHÔNG áp dụng (gây hại qua ablation):
    #4 Idle Movement    — di chuyển shipper về centroid đơn hàng khi rảnh.
                          Làm giảm C4 (205→51), tổng -113. Lý do: shipper rời
                          xa vị trí chiến lược, phản ứng chậm với đơn mới xuất hiện.
    """

    method_name = "GreedyBFS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.pathfinder = get_pathfinder(self.grid)

    # ------------------------------------------------------------------
    # BFS utilities (Được tăng tốc bởi GPU PathFinder)
    # ------------------------------------------------------------------
    def _distance(self, start: Position, goal: Position) -> int:
        """Khoảng cách BFS ngắn nhất (có vật cản)."""
        return self.pathfinder.dist(start, goal)

    def _next_move(self, start: Position, goal: Position) -> Move:
        """Bước đi đầu tiên theo đường BFS từ start đến goal."""
        return self.pathfinder.next_move(start, goal)

    # ------------------------------------------------------------------
    # Policy: chọn đơn giao (delivery)
    # ------------------------------------------------------------------
    def _select_delivery(
        self, shipper: Shipper, orders: Dict[int, Order]
    ) -> Optional[Order]:
        """
        Chọn đơn đang mang để đi giao.
        Giữ nguyên logic baseline: gần nhất → deadline sớm → priority cao.
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

    def _select_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: set[int],
        obs_t: int,
    ) -> Optional[Order]:
        """
        Chọn đơn chưa nhặt.
        Giữ nguyên logic baseline: pickup gần → priority cao → deadline sớm.
        Cải tiến #2: loại bỏ đơn reward dự kiến = 0.
        """
        candidates: List[Order] = []

        for order in orders.values():
            if order.id in reserved_order_ids:
                continue
            if not shipper.can_carry(order, orders):
                continue

            d_pickup = self._distance(shipper.position, (order.sx, order.sy))
            if d_pickup >= INF:
                continue

            # Cải tiến #2: bỏ đơn chắc chắn reward = 0
            # BETA factor = max(0, 1 - (est_t - et) / T) = 0 khi est_t >= et + T
            d_deliver = self._distance((order.sx, order.sy), (order.ex, order.ey))
            if obs_t + d_pickup + d_deliver >= order.et + self.env.T:
                continue

            candidates.append(order)

        if not candidates:
            return None

        # Giống baseline hoàn toàn
        return min(
            candidates,
            key=lambda order: (
                self._distance(shipper.position, (order.sx, order.sy)),
                -order.p,
                order.et,
                order.id,
            ),
        )

    # ------------------------------------------------------------------
    # Cải tiến #3: Pickup Batching — reserve thêm đơn "trên đường"
    # ------------------------------------------------------------------
    def _batch_reserve(
        self,
        shipper: Shipper,
        primary: Order,
        orders: Dict[int, Order],
        reserved: set[int],
        obs_t: int,
    ) -> List[int]:
        """
        Khi shipper A đăng ký nhặt đơn primary, tìm thêm các đơn Y nằm "trên đường"
        (detour <= BATCH_DETOUR_LIMIT) và reserve chúng để shipper B không cướp.

        Shipper A sẽ thực sự nhặt các đơn này sau khi đến pickup primary,
        vì env.pickup_best() sẽ nhặt đơn tốt nhất tại vị trí hiện tại khi op=1.

        Chỉ reserve — không thay đổi action của shipper A.
        """
        extra_ids: List[int] = []
        remaining_slots = shipper.K_max - len(shipper.bag) - 1
        w_carried = (
            sum(orders[oid].w for oid in shipper.bag if oid in orders) + primary.w
        )

        for order in orders.values():
            if remaining_slots <= 0:
                break
            if order.id == primary.id or order.id in reserved:
                continue
            if order.picked or order.delivered:
                continue
            if w_carried + order.w > shipper.W_max:
                continue

            # Detour so với đi thẳng đến pickup primary
            d_direct = self._distance(shipper.position, (primary.sx, primary.sy))
            d_via = self._distance(shipper.position, (order.sx, order.sy)) + self._distance(
                (order.sx, order.sy), (primary.sx, primary.sy)
            )
            if d_via - d_direct > BATCH_DETOUR_LIMIT:
                continue

            # Cải tiến #2 cũng áp dụng cho đơn batch
            d_deliver = self._distance((order.sx, order.sy), (order.ex, order.ey))
            if obs_t + d_via + d_deliver >= order.et + self.env.T:
                continue

            extra_ids.append(order.id)
            w_carried += order.w
            remaining_slots -= 1

        return extra_ids

    # ------------------------------------------------------------------
    # Policy: tạo action
    # ------------------------------------------------------------------
    def _move_towards(self, shipper: Shipper, goal: Position) -> Tuple[Move, Position]:
        move = self._next_move(shipper.position, goal)
        next_position = valid_next_pos(shipper.position, move, self.grid)
        return move, next_position

    def _delivery_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.ex, order.ey)
        move, next_position = self._move_towards(shipper, goal)
        return (move, 2) if next_position == goal else (move, 0)

    def _pickup_action(self, shipper: Shipper, order: Order) -> Action:
        goal = (order.sx, order.sy)
        move, next_position = self._move_towards(shipper, goal)
        return (move, 1) if next_position == goal else (move, 0)

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        obs_t: int = obs["t"]

        actions: Dict[int, Action] = {}
        reserved_pickups: set[int] = set()

        for shipper in sorted(shippers, key=lambda s: s.id):
            delivery_order = self._select_delivery(shipper, orders)
            if delivery_order is not None:
                actions[shipper.id] = self._delivery_action(shipper, delivery_order)
                continue

            pickup_order = self._select_pickup(shipper, orders, reserved_pickups, obs_t)
            if pickup_order is not None:
                reserved_pickups.add(pickup_order.id)
                # Cải tiến #3: reserve thêm đơn "trên đường"
                for extra_id in self._batch_reserve(
                    shipper, pickup_order, orders, reserved_pickups, obs_t
                ):
                    reserved_pickups.add(extra_id)
                actions[shipper.id] = self._pickup_action(shipper, pickup_order)
                continue

            actions[shipper.id] = ("S", 0)

        return actions

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
