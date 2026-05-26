from __future__ import annotations

import time
from collections import deque, OrderedDict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos, delivery_reward
from solvers.solver import Solver
from solvers.pathfinder import get_pathfinder
from solvers.shared.detector import OnlineSurgeHotspotDetector


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
        # Thiết lập max_delivery_delay động theo kích thước bản đồ
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
        # Tiền tính toán adjacency list
        self._adj: Dict[Position, List[Tuple[Move, Position]]] = {}
        self._build_adjacency_list()
        
        # Single-source BFS cache: start -> (dist_map, next_move_map or None)
        self._bfs_cache: Dict[Position, Tuple[Dict[Position, int], Optional[Dict[Position, Move]]]] = {}
        self.pathfinder = get_pathfinder(self.grid)
        pathfinder_device = str(getattr(self.pathfinder, "device", "cpu"))
        self._use_gpu_pathfinder = bool(
            getattr(self.pathfinder, "has_torch", False)
            and (pathfinder_device.startswith("cuda") or pathfinder_device == "mps")
        )
        
        # Cache khoảng cách từ (sx, sy) đến (ex, ey) của từng đơn hàng: order_id -> khoảng cách
        self._order_delivery_dist: Dict[int, int] = {}
        
        self.detector = OnlineSurgeHotspotDetector(
            N=self.env.N,
            C=self.env.C,
            G=self.env.G,
            T=self.env.T,
            grid=self.grid
        )
        self._seen_order_ids: Set[int] = set()

    def _get_order_delivery_dist(self, order: Order) -> int:
        """Trả về khoảng cách từ pickup đến delivery của đơn hàng O(1) từ cache hoặc BFS."""
        if order.id in self._order_delivery_dist:
            return self._order_delivery_dist[order.id]
        dist = self._distance((order.sx, order.sy), (order.ex, order.ey))
        self._order_delivery_dist[order.id] = dist
        return dist

    def _build_adjacency_list(self):
        """Xây dựng danh sách kề cho tất cả các ô trống hợp lệ trên bản đồ."""
        N = len(self.grid)
        for r in range(N):
            for c in range(N):
                pos = (r, c)
                if is_valid_cell(pos, self.grid):
                    neighbors = []
                    for move in MOVES:
                        nxt = valid_next_pos(pos, move, self.grid)
                        if nxt != pos:
                            neighbors.append((move, nxt))
                    self._adj[pos] = neighbors

    def _neighbors(self, pos: Position) -> List[Tuple[Move, Position]]:
        """Trả về danh sách láng giềng kề hợp lệ O(1) từ cache."""
        return self._adj.get(pos, [])

    def _save_bfs_cache(self, key: Position, val: Tuple[Dict[Position, int], Optional[Dict[Position, Move]]]):
        if len(self._bfs_cache) > 4000:
            self._bfs_cache.clear()
        self._bfs_cache[key] = val

    def _gpu_dist_payload(
        self,
        start: Position,
        include_next_moves: bool,
    ) -> Optional[Tuple[Dict[Position, int], Optional[Dict[Position, Move]]]]:
        if not self._use_gpu_pathfinder:
            return None
        if hasattr(self.pathfinder, "dist_payload"):
            return self.pathfinder.dist_payload(start, include_next_moves)

        start_idx = self.pathfinder.cell_to_idx.get(start)
        if start_idx is None:
            return {start: 0}, ({start: "S"} if include_next_moves else None)

        row = self.pathfinder.dist_matrix[start_idx]
        dist_map: Dict[Position, int] = {}
        for idx, raw_dist in enumerate(row):
            dist = int(raw_dist)
            if dist < 9999:
                dist_map[self.pathfinder.idx_to_cell[idx]] = dist

        if not include_next_moves:
            return dist_map, None

        move_names = ("U", "D", "L", "R")
        move_row = self.pathfinder.next_move_matrix[start_idx]
        next_move_map: Dict[Position, Move] = {start: "S"}
        for idx, raw_move in enumerate(move_row):
            if row[idx] >= 9999:
                continue
            move_idx = int(raw_move)
            next_move_map[self.pathfinder.idx_to_cell[idx]] = (
                move_names[move_idx] if 0 <= move_idx < 4 else "S"
            )
        return dist_map, next_move_map

    def _bfs_from(self, start: Position) -> Tuple[Dict[Position, int], Dict[Position, Move]]:
        """Chạy BFS một nguồn (start) tính khoảng cách và next move tới tất cả các ô có thể đi đến."""
        if start in self._bfs_cache and self._bfs_cache[start][1] is not None:
            return self._bfs_cache[start]

        gpu_payload = self._gpu_dist_payload(start, include_next_moves=True)
        if gpu_payload is not None:
            dist_map, next_move_map = gpu_payload
            if next_move_map is None:
                next_move_map = {start: "S"}
            self._save_bfs_cache(start, (dist_map, next_move_map))
            return dist_map, next_move_map

        dist_map = {start: 0}
        next_move_map = {start: "S"}

        if not is_valid_cell(start, self.grid):
            self._save_bfs_cache(start, (dist_map, next_move_map))
            return dist_map, next_move_map

        queue = deque()
        queue_append = queue.append
        queue_popleft = queue.popleft
        neighbors = self._adj
        
        for move, nxt in neighbors.get(start, []):
            dist_map[nxt] = 1
            next_move_map[nxt] = move
            queue_append(nxt)

        while queue:
            curr = queue_popleft()
            d_curr = dist_map[curr]
            m_curr = next_move_map[curr]
            d_nxt = d_curr + 1

            for _, nxt in neighbors.get(curr, []):
                if nxt not in dist_map:
                    dist_map[nxt] = d_nxt
                    next_move_map[nxt] = m_curr
                    queue_append(nxt)

        self._save_bfs_cache(start, (dist_map, next_move_map))
        return dist_map, next_move_map

    def _dist_bfs_from(self, start: Position) -> Dict[Position, int]:
        """Chạy BFS chỉ tính khoảng cách (không tính next move) để tối ưu hiệu năng."""
        if start in self._bfs_cache:
            return self._bfs_cache[start][0]

        gpu_payload = self._gpu_dist_payload(start, include_next_moves=False)
        if gpu_payload is not None:
            dist_map, _ = gpu_payload
            self._save_bfs_cache(start, (dist_map, None))
            return dist_map

        # Nếu start là vị trí của shipper, ta tính gộp cả next_move_map
        if hasattr(self, "_all_shipper_positions") and start in self._all_shipper_positions:
            dist_map, _ = self._bfs_from(start)
            return dist_map

        dist_map = {start: 0}
        if not is_valid_cell(start, self.grid):
            self._save_bfs_cache(start, (dist_map, None))
            return dist_map

        queue = deque([start])
        queue_append = queue.append
        queue_popleft = queue.popleft
        neighbors = self._adj

        while queue:
            curr = queue_popleft()
            d_nxt = dist_map[curr] + 1

            for _, nxt in neighbors.get(curr, []):
                if nxt not in dist_map:
                    dist_map[nxt] = d_nxt
                    queue_append(nxt)

        self._save_bfs_cache(start, (dist_map, None))
        return dist_map

    def _distance(self, start: Position, goal: Position) -> int:
        """Khoảng cách đường đi ngắn nhất giữa start và goal."""
        if start == goal:
            return 0
        if start in self._bfs_cache:
            return self._bfs_cache[start][0].get(goal, INF)
        if goal in self._bfs_cache:
            return self._bfs_cache[goal][0].get(start, INF)
            
        dist_map = self._dist_bfs_from(start)
        return dist_map.get(goal, INF)

    def _quick_distance(self, start: Position, goal: Position) -> int:
        """Khoảng cách nhanh sử dụng cache BFS, nếu chưa tính thì ước lượng bằng Manhattan."""
        if start == goal:
            return 0
        if start in self._bfs_cache:
            return self._bfs_cache[start][0].get(goal, INF)
        if goal in self._bfs_cache:
            return self._bfs_cache[goal][0].get(start, INF)
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    def _next_move(self, start: Position, goal: Position) -> Move:
        """Bước đi tiếp theo đầu tiên từ start đi đến goal."""
        if start == goal:
            return "S"
        if start in self._bfs_cache and self._bfs_cache[start][1] is not None:
            return self._bfs_cache[start][1].get(goal, "S")
        _, next_move_map = self._bfs_from(start)
        return next_move_map.get(goal, "S")

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

        current_t = self.env.t
        def sort_key(order):
            dist = self._distance(shipper.position, (order.ex, order.ey))
            is_on_time = (current_t + dist <= order.et)
            if is_on_time:
                # Nhóm 0: Có thể đúng hạn. Ưu tiên: deadline sớm nhất, rồi đến gần nhất
                return (0, order.et, dist, -order.p, order.id)
            else:
                # Nhóm 1: Chắc chắn trễ hạn. Ưu tiên: gần nhất (giải phóng bag), deadline
                return (1, dist, order.et, -order.p, order.id)

        return min(carried_orders, key=sort_key)



    """Tầng 0: Reward-Aware Scoring"""
    def _order_pickup_score(self, shipper, order, current_t, T):
        """
        Ước tính net reward/step nếu shipper đi nhặt rồi giao order này.
        Giá trị càng cao = đơn càng đáng nhặt.
        """
        key = (shipper.position, order.id)
        if hasattr(self, "_score_cache") and key in self._score_cache:
            return self._score_cache[key]

        dist_pickup  = self._distance(shipper.position, (order.sx, order.sy))
        dist_deliver = self._get_order_delivery_dist(order)
        
        if dist_pickup >= INF or dist_deliver >= INF:
            score = -INF
        else:
            # Thời điểm ước tính giao được hàng
            t_estimated_delivery = current_t + dist_pickup + dist_deliver
            
            # Phần thưởng ước tính theo công thức đề bài
            expected_reward = delivery_reward(order, t_estimated_delivery, T)
            
            expiry_mult = 1.0
            
            # Tổng chi phí bước đi (dùng để normalize)
            total_steps = max(dist_pickup + dist_deliver, 1)
            
            # Urgency factor: đơn sắp hết hạn thì ưu tiên hơn, nhưng nếu trễ thì urgency = 0
            if t_estimated_delivery > order.et:
                urgency = 0.0
            else:
                time_slack = max(order.et - current_t, 1)
                urgency = 1.0 / time_slack  # cao nếu deadline gần
            
            score = (expected_reward * expiry_mult) / total_steps + urgency * 10.0

        if hasattr(self, "_score_cache"):
            self._score_cache[key] = score
        return score

    def _select_pickup_v1(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: set[int],
    ) -> Optional[Order]:
        """
        Chọn đơn chưa nhặt dựa trên ước lượng phần thưởng tối ưu nhất.
        """
        candidates: List[Order] = []

        for order in orders.values():
            if order.id in reserved_order_ids:
                continue
            if not shipper.can_carry(order, orders):
                continue
            candidates.append(order)

        if not candidates:
            return None

        # Chạy BFS để có khoảng cách chính xác tới mọi ô từ vị trí shipper
        dist_map = self._dist_bfs_from(shipper.position)

        valid_candidates = []
        for order in candidates:
            if dist_map.get((order.sx, order.sy), INF) < INF:
                valid_candidates.append(order)

        if not valid_candidates:
            return None

        return max(
            valid_candidates,
            key=lambda order: (
                self._order_pickup_score(shipper, order, self.env.t, self.env.T),
                -order.id,
            ),
        )


    """
    Chỉ nhặt khi phần thưởng tăng lên - Tăng 1 chút
    Kết quả: 2551.47
    """
    def _estimate_route_times(
        self,
        shipper_pos: Position,
        pickup_orders: List[Order],
        bag_orders: List[Order],
        start_t: int
    ) -> Dict[int, int]:
        """
        Ước lượng thời gian giao hàng thực tế theo lộ trình nhặt và giao.
        """
        t = start_t
        curr = shipper_pos
        is_large_map = (self.env.N > 50)
        
        # Phase 1: Nhặt tất cả pickup_orders
        for o in pickup_orders:
            dist = self._distance(curr, (o.sx, o.sy))
            if dist >= INF:
                return {}
            if is_large_map:
                if dist > 0:
                    t += dist
                else:
                    t += 1
            else:
                t += dist + 1
            curr = (o.sx, o.sy)
        
        # Phase 2: Giao hàng (bao gồm cả pickup_orders đã nhặt xong và các đơn đã có sẵn trong bag)
        remaining_deliveries = list(pickup_orders) + list(bag_orders)
        delivery_times = {}
        
        last_was_delivery = False
        last_delivery_pos = None
        
        while remaining_deliveries:
            best_d = min(
                remaining_deliveries,
                key=lambda o: (
                    o.et,
                    self._distance(curr, (o.ex, o.ey)),
                    -o.p,
                    o.id
                )
            )
            dist = self._distance(curr, (best_d.ex, best_d.ey))
            if dist >= INF:
                return {}
            
            if is_large_map:
                if dist > 0:
                    t += dist
                else:
                    if last_was_delivery and last_delivery_pos == (best_d.ex, best_d.ey):
                        pass
                    else:
                        t += 1
            else:
                t += dist + 1
                
            delivery_times[best_d.id] = t
            curr = (best_d.ex, best_d.ey)
            last_was_delivery = True
            last_delivery_pos = curr
            remaining_deliveries.remove(best_d)
            
        return delivery_times

    def _evaluate_opportunistic_pickup(
        self,
        shipper: Shipper,
        candidate: Order,
        current_t: int,
        orders: Dict[int, Order],
        bag_orders: List[Order],
        baseline_reward: float,
    ) -> float:
        """
        Tính net gain (reward) nếu nhặt thêm candidate.
        """
        T = self.env.T
        
        # Với candidate: đi nhặt candidate trước, rồi giao tất cả
        new_times = self._estimate_route_times(shipper.position, [candidate], bag_orders, current_t)
        if not new_times:
            return -INF
            
        # Đảm bảo việc nhặt candidate không làm trễ bất kỳ đơn nào đang có trong bag
        for o in bag_orders:
            if new_times.get(o.id, INF) > o.et:
                return -INF
        
        # Cũng không làm trễ chính candidate
        if new_times.get(candidate.id, INF) > candidate.et:
            return -INF
            
        # Tính tổng reward mới
        new_reward = 0.0
        for o in bag_orders + [candidate]:
            est_t = new_times.get(o.id, INF)
            new_reward += delivery_reward(o, est_t, T)
            
        # Trừ đi chi phí di chuyển tăng thêm do tải trọng tăng
        d_to_cpickup = self._distance(shipper.position, (candidate.sx, candidate.sy))
        w_extra = candidate.w
        extra_move_cost = d_to_cpickup * (-0.01 * w_extra / max(shipper.W_max, 1.0))
        
        net_gain = (new_reward + extra_move_cost) - baseline_reward
        return net_gain


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

        bag_orders = [
            orders[oid] for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        
        # 1. Tính baseline reward (không nhặt candidate, đi giao các đơn hiện tại)
        baseline_times = self._estimate_route_times(shipper.position, [], bag_orders, current_t)
        if not baseline_times:
            return None
        
        baseline_reward = 0.0
        for o in bag_orders:
            est_t = baseline_times.get(o.id, INF)
            baseline_reward += delivery_reward(o, est_t, self.env.T)

        # BFS để lấy khoảng cách thực tế từ shipper.position
        dist_map = self._dist_bfs_from(shipper.position)

        best_order: Optional[Order] = None
        best_gain = 0.0  # Ngưỡng: chỉ nhặt khi gain > 0

        for order in orders.values():
            if order.picked or order.delivered:
                continue
            if order.id in reserved_order_ids:
                continue
            if not shipper.can_carry(order, orders):
                continue

            # Chỉ nhặt đơn cơ hội nếu khoảng cách hợp lý
            d_to_pickup = dist_map.get((order.sx, order.sy), INF)
            max_opp_dist = 15 if self.env.N > 30 else 30
            if d_to_pickup > max_opp_dist:
                continue

            # Loại nhanh: đơn đã hết deadline không cần tính
            if order.et <= current_t:
                continue

            # Check khả thi nhanh: d_to_pickup + d_to_delivery > thời gian còn lại
            d_to_delivery = self._get_order_delivery_dist(order)
            if current_t + d_to_pickup + d_to_delivery > order.et:
                continue

            gain = self._evaluate_opportunistic_pickup(
                shipper, order, current_t, orders, bag_orders, baseline_reward
            )
            if gain > best_gain:
                best_gain  = gain
                best_order = order

        return best_order



    """Tầng 2: Pickup nhiều đơn cùng lúc khi bag trống - kết quả 2979.62"""
    def _plan_multi_pickup_route(self, shipper, available_orders, current_t):
        """
        Lập kế hoạch nhặt nhiều đơn trong một chuyến khi bag rỗng.
        """
        cache_key = (shipper.id, frozenset(available_orders.keys()))
        if hasattr(self, "_multi_pickup_cache") and cache_key in self._multi_pickup_cache:
            return self._multi_pickup_cache[cache_key]
        candidates = [
            o for o in available_orders.values()
            if not o.picked and not o.delivered
        ]
        if not candidates:
            return None
        
        # Lấy khoảng cách BFS thực tế từ vị trí shipper
        dist_map = self._dist_bfs_from(shipper.position)
        
        # Lọc các đơn thực sự đến được
        candidates = [o for o in candidates if dist_map.get((o.sx, o.sy), INF) < INF]
        if not candidates:
            return None
            
        # Sắp xếp các đơn theo khoảng cách BFS tăng dần từ vị trí shipper
        candidates.sort(key=lambda o: dist_map.get((o.sx, o.sy), INF))
        
        # Greedy nearest-neighbor
        route, total_weight, total_slots = [], 0.0, 0
        current_pos = shipper.position
        remaining = list(candidates)
        curr_t = current_t
        
        limit = self.max_delivery_delay

        while remaining and total_slots < shipper.K_max:
            # Lấy khoảng cách từ current_pos hiện tại
            curr_dist = self._dist_bfs_from(current_pos)
            
            # Lọc và sắp xếp các đơn còn lại có thể đi đến được và thỏa mãn trọng tải/slot
            valid_rem = [o for o in remaining 
                         if total_weight + o.w <= shipper.W_max 
                         and total_slots + 1 <= shipper.K_max
                         and curr_dist.get((o.sx, o.sy), INF) < INF]
            if not valid_rem:
                break
                
            # Sắp xếp theo khoảng cách từ current_pos
            valid_rem.sort(key=lambda o: curr_dist.get((o.sx, o.sy), INF))
            
            found_next = False
            for best in valid_rem:
                # Quick feasibility check
                d_to_pickup = curr_dist.get((best.sx, best.sy), INF)
                d_to_delivery = self._get_order_delivery_dist(best)
                if curr_t + d_to_pickup + d_to_delivery + 2 > best.et + limit:
                    remaining.remove(best)
                    continue
                
                # Thử thêm `best` vào route tạm thời
                test_route = route + [best]
                delivery_times = self._estimate_route_times(shipper.position, test_route, [], current_t)
                
                if delivery_times:
                    # Đảm bảo các đơn trong route không bị trễ quá limit
                    ok = True
                    for o in test_route:
                        if delivery_times.get(o.id, INF) - o.et > limit:
                            ok = False
                            break
                    
                    if ok:
                        route = test_route
                        curr_t += d_to_pickup
                        current_pos = (best.sx, best.sy)
                        total_weight += best.w
                        total_slots += 1
                        remaining.remove(best)
                        found_next = True
                        break  # Thoát loop inner để cập nhật current_pos và curr_dist
                
                # Nếu không thể dùng best, loại nó khỏi remaining
                remaining.remove(best)
            
            if not found_next:
                # Nếu duyệt qua tất cả valid_rem mà không thêm được đơn nào, dừng lại
                break
        
        res = route[0] if route else None
        if hasattr(self, "_multi_pickup_cache"):
            self._multi_pickup_cache[cache_key] = res
        return res




    def _is_bottleneck(self, pos: Position) -> bool:
        """Kiểm tra ô pos có phải nút cổ chai (<= 2 ô trống xung quanh) hay không."""
        r, c = pos
        free_neighbors = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < len(self.grid) and 0 <= nc < len(self.grid[0]) and self.grid[nr][nc] == 0:
                free_neighbors += 1
        return free_neighbors <= 2

    def _bfs_path_avoiding(self, start: Position, goal: Position, other_positions: Set[Position]) -> Optional[List[Move]]:
        if start == goal:
            return []
            
        queue = deque([start])
        visited = {start}
        parent = {}
        neighbors = self._adj
        
        found = False
        while queue:
            curr = queue.popleft()
            if curr == goal:
                found = True
                break
                
            curr_neighbors = neighbors.get(curr, [])
            for move, nxt in curr_neighbors:
                if nxt in visited:
                    continue
                
                # Tránh các ô có shipper khác đang ở trong nút cổ chai
                if nxt != goal and nxt in other_positions and self._is_bottleneck(nxt):
                    continue
                    
                visited.add(nxt)
                parent[nxt] = (curr, move)
                queue.append(nxt)
                
        if not found:
            return None

        # Reconstruct path
        path = []
        curr = goal
        while curr != start:
            curr, move = parent[curr]
            path.append(move)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Policy: tạo action
    # ------------------------------------------------------------------
    def _move_towards(self, shipper: Shipper, goal: Position) -> Tuple[Move, Position]:
        """
        Lấy bước đi kế tiếp và vị trí dự kiến sau bước đó.
        """
        start = shipper.position
        if start == goal:
            return "S", start

        move = self._next_move(start, goal)
        
        # Chỉ tránh nút cổ chai trên bản đồ lớn (N >= 100)
        if len(self.grid) >= 100:
            blocking_obstacles = set()
            shipper_goals = getattr(self, "_shipper_goals", {})
            all_shippers = getattr(self, "_all_shippers", [])
            
            for s in all_shippers:
                if s.id == shipper.id:
                    continue
                pos_B = s.position
                if self._is_bottleneck(pos_B):
                    # Ước lượng điểm đến của B
                    goal_B = shipper_goals.get(s.id, pos_B)
                    move_B = self._next_move(pos_B, goal_B)
                    nxt_B = valid_next_pos(pos_B, move_B, self.grid)
                    
                    # Kiểm tra xem B có đang đi về phía A (ngược chiều) hoặc đứng yên hay không
                    dist_pos_B = abs(pos_B[0] - start[0]) + abs(pos_B[1] - start[1])
                    dist_nxt_B = abs(nxt_B[0] - start[0]) + abs(nxt_B[1] - start[1])
                    
                    # B không đi xa A ra (tức là đi ngược chiều hoặc đứng yên)
                    if dist_nxt_B <= dist_pos_B:
                        # Nếu là hàng xóm trực tiếp (khoảng cách = 1) và đối đầu trực diện:
                        # Giải quyết bằng độ ưu tiên ID (ID nhỏ được đi, ID lớn nhường)
                        if dist_pos_B == 1:
                            if shipper.id > s.id:
                                blocking_obstacles.add(pos_B)
                        else:
                            blocking_obstacles.add(pos_B)
                            
            if blocking_obstacles:
                # Kiểm tra xem đường đi chuẩn 15 bước tới có đi qua ô bị chặn nào không
                path_blocked = False
                curr = start
                path_set = {curr}
                for _ in range(15):
                    move_step = self._next_move(curr, goal)
                    if move_step == "S":
                        break
                    nxt = valid_next_pos(curr, move_step, self.grid)
                    if nxt == curr or nxt in path_set:
                        break
                    path_set.add(nxt)
                    if nxt in blocking_obstacles:
                        path_blocked = True
                        break
                    curr = nxt

                if path_blocked:
                    alt_path = self._bfs_path_avoiding(start, goal, blocking_obstacles)
                    if alt_path:
                        move = alt_path[0]
                    else:
                        move = "S"  # Đứng yên ngoài nút cổ chai chờ thông đường

        next_position = valid_next_pos(start, move, self.grid)
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

    def _reposition_action(self, shipper: Shipper, orders: Dict[int, Order], current_t: int) -> Tuple[Action, Position]:
        """
        Di chuyển shipper trống về phía predicted hotspots khi có surge.
        """
        # 1. Chỉ reposition khi có surge và hotspots
        if hasattr(self, "detector") and self.detector.is_surge and self.detector.predicted_hotspots:
            # Chọn hotspot gần shipper nhất
            best_hotspot = min(
                self.detector.predicted_hotspots,
                key=lambda hp: self._distance(shipper.position, hp)
            )
            if best_hotspot != shipper.position:
                move, next_pos = self._move_towards(shipper, best_hotspot)
                return (move, 0), best_hotspot
                
        # 2. Nếu không có surge hoặc hotspot, đứng yên để tránh lãng phí move cost
        return ("S", 0), shipper.position

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        self._score_cache = {}
        self._multi_pickup_cache = {}
        orders   = obs["orders"]
        shippers = obs["shippers"]
        current_t = obs.get("t", 0)
        
        # Cập nhật detector
        current_order_ids = set(orders.keys())
        new_order_ids = list(current_order_ids - self._seen_order_ids)
        self._seen_order_ids.update(current_order_ids)
        self.detector.update(current_t, new_order_ids, orders)
        
        self._all_shippers = shippers
        self._all_shipper_positions = {s.position for s in shippers}
        
        # Tiền tính toán mục tiêu của mỗi shipper để ước lượng hướng đi của chúng
        self._shipper_goals = {}
        for s in shippers:
            if len(s.bag) > 0:
                delivery_order = self._select_delivery(s, orders)
                if delivery_order:
                    self._shipper_goals[s.id] = (delivery_order.ex, delivery_order.ey)
                else:
                    self._shipper_goals[s.id] = s.position
            else:
                pickup_order = self._plan_multi_pickup_route(s, orders, current_t)
                if pickup_order is None:
                    # heuristic: chọn đơn chưa nhận gần nhất
                    best_order = None
                    best_dist = INF
                    for o in orders.values():
                        if not o.picked and not o.delivered:
                            d = abs(s.r - o.sx) + abs(s.c - o.sy)
                            if d < best_dist:
                                  best_dist = d
                                  best_order = o
                    if best_order:
                        pickup_order = best_order
                if pickup_order:
                    self._shipper_goals[s.id] = (pickup_order.sx, pickup_order.sy)
                else:
                    self._shipper_goals[s.id] = s.position

        actions: Dict[int, Action]   = {}
        goals: Dict[int, Position]   = {}
        reserved_pickups: set[int]   = set()
        self._last_types = {}

        shippers_with_cargo = []
        shippers_empty = []
        for shipper in shippers:
            if len(shipper.bag) > 0:
                shippers_with_cargo.append(shipper)
            else:
                shippers_empty.append(shipper)

        # 1. Khớp cặp shipper trống và đơn hàng bằng cơ chế Greedy Matching (phối hợp toàn cục)
        unmatched_shippers = list(shippers_empty)
        while unmatched_shippers:
            candidates_for_shippers = []
            for shipper in unmatched_shippers:
                available_orders = {oid: o for oid, o in orders.items() if oid not in reserved_pickups}
                pickup_order = self._plan_multi_pickup_route(shipper, available_orders, current_t)
                if pickup_order is None:
                    pickup_order = self._select_pickup_v1(shipper, orders, reserved_pickups)
                
                if pickup_order is not None:
                    dist = self._distance(shipper.position, (pickup_order.sx, pickup_order.sy))
                    score = self._order_pickup_score(shipper, pickup_order, current_t, self.env.T)
                    candidates_for_shippers.append((score, dist, shipper, pickup_order))
            
            if not candidates_for_shippers:
                # Các shipper còn lại không tìm được đơn hàng nào phù hợp
                # Phân công hotspot theo kiểu Round-robin
                hotspots = self.detector.predicted_hotspots if (hasattr(self, "detector") and self.detector.is_surge and self.detector.predicted_hotspots) else []
                sorted_unmatched = sorted(unmatched_shippers, key=lambda s: s.id)
                
                for idx, shipper in enumerate(sorted_unmatched):
                    if hotspots:
                        assigned_hotspot = hotspots[idx % len(hotspots)]
                        if assigned_hotspot != shipper.position:
                            move, next_pos = self._move_towards(shipper, assigned_hotspot)
                            actions[shipper.id] = (move, 0)
                            goals[shipper.id] = assigned_hotspot
                            self._last_types[shipper.id] = "reposition" if move != "S" else "none"
                            continue
                    
                    # Nếu không có hotspot hoặc đã ở đó
                    actions[shipper.id] = ("S", 0)
                    goals[shipper.id] = shipper.position
                    self._last_types[shipper.id] = "none"
                break
                
            # Sắp xếp theo score giảm dần (ưu tiên đơn mang lại reward/step cao nhất), sau đó là dist tăng dần
            candidates_for_shippers.sort(key=lambda x: (-x[0], x[1]))
            
            # Chọn cặp khớp tốt nhất ở bước này
            best_score, best_dist, best_shipper, best_order = candidates_for_shippers[0]
            
            reserved_pickups.add(best_order.id)
            actions[best_shipper.id] = self._pickup_action(best_shipper, best_order)
            goals[best_shipper.id] = (best_order.sx, best_order.sy)
            self._last_types[best_shipper.id] = "pickup"
            unmatched_shippers.remove(best_shipper)

        # 2. Xử lý shippers đang mang hàng sau
        for shipper in sorted(shippers_with_cargo, key=lambda s: (len(s.bag), s.id)):
            # Nếu shipper đang đứng tại đích của bất kỳ đơn nào trong bag, giao ngay lập tức!
            at_delivery_dest = False
            for oid in shipper.bag:
                if oid in orders and not orders[oid].delivered:
                    o = orders[oid]
                    if shipper.position == (o.ex, o.ey):
                        at_delivery_dest = True
                        break
            if at_delivery_dest:
                actions[shipper.id] = ("S", 2)
                goals[shipper.id] = shipper.position
                self._last_types[shipper.id] = "delivery"
                continue

            delivery_order = self._select_delivery(shipper, orders)

            if delivery_order is not None:
                # Không làm opportunistic pickup cho C4 để ổn định điểm số
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
                        goals[shipper.id] = shipper.position
                        self._last_types[shipper.id] = "pickup"
                    else:
                        dist_to_opp_pickup = self._distance(
                            shipper.position, (opp.sx, opp.sy)
                        )
                        dist_to_delivery = self._distance(
                            shipper.position, (delivery_order.ex, delivery_order.ey)
                        )
                        if dist_to_opp_pickup <= dist_to_delivery:
                            actions[shipper.id] = self._pickup_action(shipper, opp)
                            goals[shipper.id] = (opp.sx, opp.sy)
                            self._last_types[shipper.id] = "pickup"
                        else:
                            actions[shipper.id] = self._delivery_action(
                                shipper, delivery_order
                            )
                            goals[shipper.id] = (delivery_order.ex, delivery_order.ey)
                            self._last_types[shipper.id] = "delivery"
                    continue

                actions[shipper.id] = self._delivery_action(shipper, delivery_order)
                goals[shipper.id] = (delivery_order.ex, delivery_order.ey)
                self._last_types[shipper.id] = "delivery"
                continue
            else:
                actions[shipper.id] = ("S", 0)
                goals[shipper.id] = shipper.position
                self._last_types[shipper.id] = "none"
        
        self._last_goals = goals
        return self._resolve_deadlocks(shippers, actions, goals)

    def _resolve_deadlocks(self, shippers: List[Shipper], actions: Dict[int, Action], goals: Dict[int, Position]) -> Dict[int, Action]:
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

        def short_path_cells(start: Position, goal: Position, limit: int = 8) -> Set[Position]:
            cells: Set[Position] = set()
            curr = start
            seen = {curr}
            for _ in range(limit):
                mv = self._next_move(curr, goal)
                if mv == "S":
                    break
                nxt = valid_next_pos(curr, mv, self.grid)
                if nxt == curr or nxt in seen:
                    break
                cells.add(nxt)
                seen.add(nxt)
                curr = nxt
                if curr == goal:
                    break
            return cells

        def find_yield_move(blocker: Shipper, mover: Shipper) -> Optional[Move]:
            blocker_pos = blocker.position
            occupied = set(positions.values())
            reserved_next = {
                pos
                for sid, pos in desired.items()
                if sid not in {blocker.id, mover.id}
            }
            mover_goal = goals.get(mover.id, mover.position)
            avoid_path = short_path_cells(mover.position, mover_goal)

            candidates: List[Tuple[int, int, Move]] = []
            for mv in MOVES:
                nxt = valid_next_pos(blocker_pos, mv, self.grid)
                if nxt == blocker_pos:
                    continue
                if nxt in occupied or nxt in reserved_next:
                    continue
                if nxt == mover.position or nxt == desired.get(mover.id):
                    continue

                # Prefer stepping out of the mover's route, then farther from mover.
                on_mover_path = 1 if nxt in avoid_path else 0
                dist_from_mover = abs(nxt[0] - mover.position[0]) + abs(nxt[1] - mover.position[1])
                candidates.append((on_mover_path, -dist_from_mover, mv))

            if not candidates:
                return None
            candidates.sort()
            return candidates[0][2]

        if self.env.N <= 25 and self.env.C <= 3:
            # Idle blocker displacement:
            # nếu shipper đang làm việc bị chặn bởi shipper rảnh đang đứng yên,
            # yêu cầu shipper rảnh né khỏi hành lang. Với env tuần tự theo id,
            # shipper bị chặn có thể vẫn phải chờ 1 tick nếu nó có id nhỏ hơn blocker,
            # nhưng tránh được kẹt nhiều tick cho tới khi đơn mới xuất hiện.
            for mover in sorted(shippers, key=lambda s: (len(s.bag) == 0, s.id)):
                mover_action = resolved.get(mover.id, ("S", 0))
                mover_target = desired.get(mover.id, mover.position)
                if mover_target == mover.position:
                    continue

                for blocker in sorted(shippers, key=lambda s: s.id):
                    if blocker.id == mover.id:
                        continue
                    if positions[blocker.id] != mover_target:
                        continue

                    blocker_action = resolved.get(blocker.id, ("S", 0))
                    blocker_target = desired.get(blocker.id, blocker.position)
                    blocker_idle = (
                        len(blocker.bag) == 0
                        and blocker_action[0] == "S"
                        and blocker_action[1] == 0
                        and blocker_target == blocker.position
                    )
                    mover_has_work = len(mover.bag) > 0 or mover_action[1] in {1, 2}
                    if not blocker_idle or not mover_has_work:
                        continue

                    yield_move = find_yield_move(blocker, mover)
                    if yield_move is None:
                        continue

                    resolved[blocker.id] = (yield_move, 0)
                    desired[blocker.id] = valid_next_pos(blocker.position, yield_move, self.grid)
                    break

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
        seen_order_ids = set()

        while not obs.get("done", False):
            for oid in obs["orders"].keys():
                seen_order_ids.add(oid)

            # Ba điều kiện kết hợp để chắc chắn không bỏ sót việc gì:
            # 1. Đã sinh đủ G đơn
            cond1 = (len(seen_order_ids) == obs["G"])
            # 2. Không shipper nào còn hàng trong tay
            cond2 = all(len(s.bag) == 0 for s in obs["shippers"])
            # 3. Không còn đơn active nào trên map (đã được pick)
            cond3 = all(o.picked for o in obs["orders"].values())

            if cond1 and cond2 and cond3:
                break

            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
