from __future__ import annotations

import random
import time
from collections import deque, OrderedDict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos, delivery_reward
from solvers.solver import Solver
from solvers.shared.detector import OnlineSurgeHotspotDetector
from solvers.shared.precompute import get_precompute

# Hashing representation of grid to cache PathFinder instances
_PATHFINDER_CACHE: Dict[Tuple[Tuple[int, ...], ...], PathFinder] = {}


def get_pathfinder(grid: List[List[int]]) -> PathFinder:
    grid_hash = tuple(tuple(row) for row in grid)
    if grid_hash not in _PATHFINDER_CACHE:
        _PATHFINDER_CACHE[grid_hash] = PathFinder(grid)
    return _PATHFINDER_CACHE[grid_hash]


class PathFinder:
    """
    Global Pathfinder that precomputes All-Pairs Shortest Path (APSP) and next-step routing
    on GPU (or CPU) using PyTorch parallel BFS when available. Falls back to dynamic CPU-based
    BFS when PyTorch is not available.
    """

    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.N = len(grid)
        self.precompute = get_precompute(grid)
        
        # Fallback CPU BFS caches (always initialized for safety/testing)
        self._bfs_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, List[str]]] = {}
        self._distance_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
        self._next_move_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], str] = {}
        
        # Fast single source CPU BFS caches
        self._adj: Dict[Tuple[int, int], List[Tuple[str, Tuple[int, int]]]] = {}
        self._single_source_cache: Dict[Tuple[int, int], Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], Tuple[Tuple[int, int], str]]]] = {}
        self._build_adjacency_list()
        
        self.has_torch = self._precompute_with_torch(grid)

    def _build_adjacency_list(self):
        for r in range(self.N):
            for c in range(self.N):
                pos = (r, c)
                if is_valid_cell(pos, self.grid):
                    neighbors = []
                    for move in ("U", "D", "L", "R"):
                        nxt = valid_next_pos(pos, move, self.grid)
                        if nxt != pos:
                            neighbors.append((move, nxt))
                    self._adj[pos] = neighbors

    def _compute_single_source(self, start: Tuple[int, int]):
        if start in self._single_source_cache:
            return
        dist_map = {start: 0}
        parent_map = {}  # nxt -> (curr, move)
        if not is_valid_cell(start, self.grid):
            self._single_source_cache[start] = (dist_map, parent_map)
            return
        queue = deque([start])
        queue_append = queue.append
        queue_popleft = queue.popleft
        neighbors = self._adj
        while queue:
            curr = queue_popleft()
            d_nxt = dist_map[curr] + 1
            for move, nxt in neighbors.get(curr, []):
                if nxt not in dist_map:
                    dist_map[nxt] = d_nxt
                    parent_map[nxt] = (curr, move)
                    queue_append(nxt)
        self._single_source_cache[start] = (dist_map, parent_map)

    def _precompute_with_torch(self, grid: List[List[int]]) -> bool:
        try:
            import torch
            
            # Use GPU if available, else CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            N = self.N
            
            # 1. Filter free cells and map them to 1D index
            free_cells = [(r, c) for r in range(N) for c in range(N) if grid[r][c] == 0]
            V = len(free_cells)
            self.cell_to_idx = {cell: i for i, cell in enumerate(free_cells)}
            self.idx_to_cell = free_cells
            
            # 2. Build adjacency tensor of shape (V+1, 4)
            # Directions: 0: U, 1: D, 2: L, 3: R. Index V is used as dummy/padding
            adj = torch.full((V + 1, 4), V, dtype=torch.long, device=device)
            moves_offset = [(-1, 0), (1, 0), (0, -1), (0, 1)]

            for idx, (r, c) in enumerate(free_cells):
                for dir_idx, (dr, dc) in enumerate(moves_offset):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] == 0:
                        adj[idx, dir_idx] = self.cell_to_idx[(nr, nc)]

            # 3. Initialize distance, first-step, visited, and frontier matrices (all shape V+1 x V+1)
            dist = torch.full((V + 1, V + 1), 9999, dtype=torch.int16, device=device)
            first_move = torch.full((V + 1, V + 1), 4, dtype=torch.int8, device=device) # 4 is Stay ('S')

            # Diagonal distance is 0 for all nodes (including dummy node V)
            diag_indices = torch.arange(V + 1, device=device)
            dist[diag_indices, diag_indices] = 0

            # Visited and frontier initialization
            visited = torch.zeros((V + 1, V + 1), dtype=torch.bool, device=device)
            visited[diag_indices, diag_indices] = True

            # Frontier initialization
            frontier = torch.zeros((V + 1, V + 1), dtype=torch.bool, device=device)
            for c in [3, 2, 1, 0]:  # Loop backwards to let smaller c (U < D < L < R) overwrite larger ones
                neighbors = adj[:, c]
                valid = neighbors < V
                if valid.any():
                    src_indices = torch.arange(V + 1, device=device)[valid]
                    dest_indices = neighbors[valid]
                    dist[src_indices, dest_indices] = 1
                    first_move[src_indices, dest_indices] = c
                    frontier[src_indices, dest_indices] = True
                    visited[src_indices, dest_indices] = True

            # 4. Multi-Source Parallel BFS loop
            for d in range(2, min(V, N * 6)):
                new_frontier = torch.zeros((V + 1, V + 1), dtype=torch.bool, device=device)
                candidates = torch.full((4, V + 1, V + 1), 127, dtype=torch.int8, device=device)
                reached_any = False
                for c in range(4):
                    reached = frontier[:, adj[:, c]] & ~visited
                    if reached.any():
                        candidates[c] = torch.where(reached, first_move[:, adj[:, c]], torch.tensor(127, dtype=torch.int8, device=device))
                        reached_any = True
                
                if not reached_any:
                    break
                
                min_candidate, _ = torch.min(candidates, dim=0)
                reached_mask = min_candidate < 127
                
                dist = torch.where(reached_mask, torch.tensor(d, dtype=torch.int16, device=device), dist)
                first_move = torch.where(reached_mask, min_candidate, first_move)
                new_frontier = reached_mask
                
                visited |= new_frontier
                frontier = new_frontier

            # 5. Extract matrices to CPU numpy for fast lookups
            self.dist_matrix = dist[:V, :V].cpu().numpy()
            self.next_move_matrix = first_move[:V, :V].cpu().numpy()
            self.adj_cpu = adj[:V].cpu().numpy()
            return True
        except Exception:
            return False

    def dist(self, start: Tuple[int, int], goal: Tuple[int, int]) -> int:
        if start == goal:
            return 0
        if not self.precompute.are_connected(start, goal):
            return 10**9
        if self.has_torch:
            s_idx = self.cell_to_idx.get(start)
            g_idx = self.cell_to_idx.get(goal)
            if s_idx is None or g_idx is None:
                return 10**9
            d = self.dist_matrix[s_idx, g_idx]
            return 10**9 if d >= 9999 else int(d)
        else:
            if start not in self._single_source_cache:
                self._compute_single_source(start)
            return self._single_source_cache[start][0].get(goal, 10**9)

    def path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[str]:
        if start == goal:
            return []
        if not self.precompute.are_connected(start, goal):
            return []
        if self.has_torch:
            s_idx = self.cell_to_idx.get(start)
            g_idx = self.cell_to_idx.get(goal)
            if s_idx is None or g_idx is None:
                return []

            path_moves = []
            curr_idx = s_idx
            moves_list = ["U", "D", "L", "R"]
            # Bounded path length to prevent infinite loops
            for _ in range(self.N * 3):
                if curr_idx == g_idx:
                    break
                move_idx = self.next_move_matrix[curr_idx, g_idx]
                if move_idx == 4 or move_idx < 0 or move_idx >= 4:
                    break
                path_moves.append(moves_list[move_idx])
                next_idx = self.adj_cpu[curr_idx, move_idx]
                if next_idx >= len(self.idx_to_cell):
                    break
                curr_idx = next_idx
            return path_moves
        else:
            if start not in self._single_source_cache:
                self._compute_single_source(start)
            _, parent_map = self._single_source_cache[start]
            if goal not in parent_map and goal != start:
                return []
            
            path_moves = []
            curr = goal
            while curr != start:
                if curr not in parent_map:
                    return []
                parent_node, move = parent_map[curr]
                path_moves.append(move)
                curr = parent_node
            path_moves.reverse()
            return path_moves

    def next_move(self, start: Tuple[int, int], goal: Tuple[int, int]) -> str:
        if start == goal:
            return "S"
        if not self.precompute.are_connected(start, goal):
            return "S"
        if self.has_torch:
            s_idx = self.cell_to_idx.get(start)
            g_idx = self.cell_to_idx.get(goal)
            if s_idx is None or g_idx is None:
                return "S"
            move_idx = self.next_move_matrix[s_idx, g_idx]
            if 0 <= move_idx < 4:
                return ["U", "D", "L", "R"][move_idx]
            return "S"
        else:
            p = self.path(start, goal)
            return p[0] if p else "S"

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
    - chọn đơn cần giao/nhặtt
    - tìm đường bằng BFS trên grid hiện tại.


    """

    method_name = "GreedyBFS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        # Sử dụng PathFinder dùng chung có tối ưu hóa GPU/CPU
        self.pathfinder = get_pathfinder(self.grid)
        self.precompute = get_precompute(self.grid)
        self._adj = self.pathfinder._adj

        # Lấy mẫu avg_dist bằng BFS
        valid_nodes = list(self._adj.keys())
        sample_size = min(len(valid_nodes), 50)
        rng = random.Random(42)
        sample_nodes = rng.sample(valid_nodes, sample_size)
        
        total_dist = 0
        pair_count = 0
        for start in sample_nodes:
            dist_map = {start: 0}
            queue = deque([start])
            queue_append = queue.append
            queue_popleft = queue.popleft
            while queue:
                curr = queue_popleft()
                d_nxt = dist_map[curr] + 1
                for _, nxt in self._adj.get(curr, []):
                    if nxt not in dist_map:
                        dist_map[nxt] = d_nxt
                        queue_append(nxt)
            for d in dist_map.values():
                if d > 0:
                    total_dist += d
                    pair_count += 1
        avg_dist = total_dist / pair_count if pair_count > 0 else 10.0

        self.avg_dist = avg_dist

        # Thiết lập max_delivery_delay động dựa trên avg_dist và bottleneck_ratio
        # Tính toán bottleneck_ratio trước
        total_free_cells = len(self._adj)
        bottleneck_cells = sum(1 for pos in self._adj.keys() if self.precompute.is_bottleneck(pos))
        self.bottleneck_ratio = bottleneck_cells / total_free_cells if total_free_cells > 0 else 0.0

        def interpolate(x, x_pts, y_pts):
            if x <= x_pts[0]:
                return y_pts[0]
            if x >= x_pts[-1]:
                return y_pts[-1]
            for i in range(len(x_pts) - 1):
                if x_pts[i] <= x <= x_pts[i+1]:
                    t = (x - x_pts[i]) / (x_pts[i+1] - x_pts[i])
                    return y_pts[i] + t * (y_pts[i+1] - y_pts[i])
            return y_pts[-1]

        x_pts = [13.66, 17.24, 17.75, 26.70, 29.95, 36.86]
        y_delay = [15, 15, 2, -5, 2, -10]
        y_opp = [15, 25, 10, 30, 20, 15]

        self.max_delivery_delay = int(round(interpolate(avg_dist, x_pts, y_delay)))
        self.max_opp_dist = int(round(interpolate(avg_dist, x_pts, y_opp)))

        self.dist_matrix = None

        # BFS cache size động
        self.max_bfs_cache_size = min(self.env.N * self.env.N, 5000)
        
        self._assignment = {}
        
        # Single-source BFS cache: start -> (dist_map, next_move_map or None)
        self._bfs_cache: OrderedDict[Position, Tuple[Dict[Position, int], Optional[Dict[Position, Move]]]] = OrderedDict()
        
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
        self.enable_reposition = (self.bottleneck_ratio < 0.3)

    def _get_order_delivery_dist(self, order: Order) -> int:
        """Trả về khoảng cách từ pickup đến delivery của đơn hàng O(1) từ cache hoặc BFS."""
        if order.id in self._order_delivery_dist:
            return self._order_delivery_dist[order.id]
        dist = self._distance((order.sx, order.sy), (order.ex, order.ey))
        self._order_delivery_dist[order.id] = dist
        return dist

    def _neighbors(self, pos: Position) -> List[Tuple[Move, Position]]:
        """Trả về danh sách láng giềng kề hợp lệ O(1) từ cache."""
        return self._adj.get(pos, [])

    def _save_bfs_cache(self, key: Position, val: Tuple[Dict[Position, int], Optional[Dict[Position, Move]]]):
        if key in self._bfs_cache:
            self._bfs_cache.pop(key)
        elif len(self._bfs_cache) >= self.max_bfs_cache_size:
            self._bfs_cache.popitem(last=False)
        self._bfs_cache[key] = val

    def _bfs_from(self, start: Position) -> Tuple[Dict[Position, int], Dict[Position, Move]]:
        """Chạy BFS một nguồn (start) tính khoảng cách và next move tới tất cả các ô có thể đi đến."""
        if start in self._bfs_cache and self._bfs_cache[start][1] is not None:
            return self._bfs_cache[start]

        if self.pathfinder.has_torch:
            s_idx = self.pathfinder.cell_to_idx.get(start)
            if s_idx is None:
                dist_map = {start: 0}
                next_move_map = {start: "S"}
            else:
                dist_map = {}
                next_move_map = {}
                moves_list = ["U", "D", "L", "R", "S"]
                # Cần tính hướng di chuyển đầu tiên từ start đến mọi đích nxt.
                # next_move_matrix[s_idx, idx] cho biết bước đi đầu tiên từ start đến cell.
                for cell, idx in self.pathfinder.cell_to_idx.items():
                    d = self.pathfinder.dist_matrix[s_idx, idx]
                    if d < 9999:
                        dist_map[cell] = int(d)
                        move_idx = self.pathfinder.next_move_matrix[s_idx, idx]
                        next_move_map[cell] = moves_list[move_idx] if move_idx < 5 else "S"
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

        if self.pathfinder.has_torch:
            s_idx = self.pathfinder.cell_to_idx.get(start)
            if s_idx is None:
                dist_map = {start: 0}
            else:
                dist_map = {}
                for cell, idx in self.pathfinder.cell_to_idx.items():
                    d = self.pathfinder.dist_matrix[s_idx, idx]
                    if d < 9999:
                        dist_map[cell] = int(d)
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
        return self.pathfinder.dist(start, goal)

    def _quick_distance(self, start: Position, goal: Position) -> int:
        """Khoảng cách nhanh sử dụng cache BFS, nếu chưa tính thì ước lượng bằng Manhattan."""
        d = self.pathfinder.dist(start, goal)
        if d < INF:
            return d
        return abs(start[0] - goal[0]) + abs(start[1] - goal[1])

    def _next_move(self, start: Position, goal: Position) -> Move:
        """Bước đi tiếp theo đầu tiên từ start đi đến goal."""
        return self.pathfinder.next_move(start, goal)

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
            if shipper.K_max <= 2:
                total_steps += dist_pickup * 0.5
            
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
            if not self.precompute.are_connected(shipper.position, (order.sx, order.sy)):
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
                if dist > 0:
                    t += dist + 1
                else:
                    if last_was_delivery and last_delivery_pos == (best_d.ex, best_d.ey):
                        pass
                    else:
                        t += 1
                
            delivery_times[best_d.id] = t
            curr = (best_d.ex, best_d.ey)
            last_was_delivery = True
            last_delivery_pos = curr
            remaining_deliveries.remove(best_d)
            
        return delivery_times

    def _estimate_route_distance(
        self,
        shipper_pos: Position,
        pickup_orders: List[Order],
        bag_orders: List[Order]
    ) -> int:
        total_dist = 0
        curr = shipper_pos
        for o in pickup_orders:
            d = self._distance(curr, (o.sx, o.sy))
            if d >= INF:
                return INF
            total_dist += d
            curr = (o.sx, o.sy)
        
        remaining = list(pickup_orders) + list(bag_orders)
        while remaining:
            best_d = min(
                remaining,
                key=lambda o: (
                    o.et,
                    self._distance(curr, (o.ex, o.ey)),
                    -o.p,
                    o.id
                )
            )
            d = self._distance(curr, (best_d.ex, best_d.ey))
            if d >= INF:
                return INF
            total_dist += d
            curr = (best_d.ex, best_d.ey)
            remaining.remove(best_d)
        return total_dist

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
            
        # Đảm bảo việc nhặt candidate không làm trễ bất kỳ đơn nào đang có trong bag và có khoảng an toàn
        safety_margin = 2 if self.env.N > 50 else 0
        for o in bag_orders:
            if new_times.get(o.id, INF) + safety_margin > o.et:
                return -INF
        
        # Cũng không làm trễ chính candidate
        if new_times.get(candidate.id, INF) + safety_margin > candidate.et:
            return -INF
            
        # Tính tổng reward mới
        new_reward = 0.0
        for o in bag_orders + [candidate]:
            est_t = new_times.get(o.id, INF)
            new_reward += delivery_reward(o, est_t, T)
            
        # Thích ứng dựa trên mật độ đơn hàng G/T
        density = self.env.G / self.env.T
        if density < 0.05:
            # Kịch bản thích ứng (mật độ thưa): Phạt detour thực tế chặt chẽ
            dist_baseline = self._estimate_route_distance(shipper.position, [], bag_orders)
            dist_new = self._estimate_route_distance(shipper.position, [candidate], bag_orders)
            if dist_baseline >= INF or dist_new >= INF:
                return -INF
            extra_steps = dist_new - dist_baseline
            extra_move_cost = -0.2 * extra_steps - (dist_new * 0.01 * candidate.w / max(shipper.W_max, 1.0))
        else:
            # Kịch bản chuẩn (mật độ dày): Không phạt detour thực tế để tối ưu hóa gom đơn
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
            density = self.env.G / self.env.T
            if density < 0.05:
                base_max_opp = min(self.max_opp_dist, max(6, int(self.env.N * 0.35)))
            else:
                base_max_opp = self.max_opp_dist
            
            # Nới rộng khi map phức tạp
            if self.avg_dist > 30:
                base_max_opp = max(base_max_opp, int(round(self.avg_dist * 1.5)))
                
            if d_to_pickup > base_max_opp:
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

        # Tiền lọc Manhattan cho map lớn/phức tạp
        should_filter = (self.env.N >= 100 or self.avg_dist > 30) and len(candidates) > 50
        if should_filter:
            candidates.sort(key=lambda o: abs(shipper.position[0] - o.sx) + abs(shipper.position[1] - o.sy))
            candidates = candidates[:60]
        
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
        return self.precompute.is_bottleneck(pos)

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

        move = self.pathfinder.next_move(start, goal)
        
        # Chỉ tránh nút cổ chai nếu tỉ lệ nút cổ chai trên bản đồ lớn (bottleneck_ratio > 0.15) hoặc bản đồ lớn (N >= 100)
        if len(self.grid) >= 100 or self.bottleneck_ratio > 0.15:
            lookahead = max(3, min(15, int(180 / self.env.C)))
            blocking_obstacles = {}  # pos_B -> is_static (bool)
            shipper_goals = getattr(self, "_shipper_goals", {})
            all_shippers = getattr(self, "_all_shippers", [])
            
            for s in all_shippers:
                if s.id == shipper.id:
                    continue
                pos_B = s.position
                if self._is_bottleneck(pos_B):
                    # Ước lượng điểm đến của B
                    goal_B = shipper_goals.get(s.id, pos_B)
                    move_B = self.pathfinder.next_move(pos_B, goal_B)
                    nxt_B = valid_next_pos(pos_B, move_B, self.grid)
                    
                    # Kiểm tra xem B có đang đi về phía A (ngược chiều) hoặc đứng yên hay không
                    dist_pos_B = abs(pos_B[0] - start[0]) + abs(pos_B[1] - start[1])
                    dist_nxt_B = abs(nxt_B[0] - start[0]) + abs(nxt_B[1] - start[1])
                    
                    # B không đi xa A ra (tức là đi ngược chiều hoặc đứng yên)
                    if dist_nxt_B <= dist_pos_B:
                        is_static = (move_B == "S")
                        blocking_obstacles[pos_B] = is_static
                            
            if blocking_obstacles:
                # Kiểm tra xem đường đi chuẩn tới có đi qua ô bị chặn nào không
                path_blocked = False
                has_static_block = False
                curr = start
                path_set = {curr}
                for _ in range(lookahead):
                    move_step = self.pathfinder.next_move(curr, goal)
                    if move_step == "S":
                        break
                    nxt = valid_next_pos(curr, move_step, self.grid)
                    if nxt == curr or nxt in path_set:
                        break
                    path_set.add(nxt)
                    if nxt in blocking_obstacles:
                        path_blocked = True
                        if blocking_obstacles[nxt]:
                            has_static_block = True
                        break
                    curr = nxt
 
                if path_blocked:
                    blocking_set = set(blocking_obstacles.keys())
                    alt_path = self._bfs_path_avoiding(start, goal, blocking_set)
                    if alt_path:
                        if has_static_block:
                            # Luôn đi đường vòng nếu có vật cản tĩnh (shipper đứng yên)
                            move = alt_path[0]
                        else:
                            # Với vật cản động, chỉ đi vòng nếu không quá xa
                            std_dist = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
                            if len(alt_path) <= std_dist + 40:
                                move = alt_path[0]
                            else:
                                move = self.pathfinder.next_move(start, goal)
                    else:
                        if has_static_block:
                            if self.env.C < 18:
                                move = "S"  # Chỉ đứng yên khi có vật cản tĩnh thực sự
                            else:
                                move = self.pathfinder.next_move(start, goal)
                        else:
                            # Không có vật cản tĩnh nào chặn, tiếp tục đi theo đường chuẩn
                            move = self.pathfinder.next_move(start, goal)

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
                available_orders = {
                    oid: o for oid, o in orders.items()
                    if oid not in reserved_pickups and self.precompute.are_connected(shipper.position, (o.sx, o.sy))
                }
                pickup_order = self._plan_multi_pickup_route(shipper, available_orders, current_t)
                if pickup_order is None:
                    pickup_order = self._select_pickup_v1(shipper, orders, reserved_pickups)
                
                if pickup_order is not None:
                    dist = self._distance(shipper.position, (pickup_order.sx, pickup_order.sy))
                    score = self._order_pickup_score(shipper, pickup_order, current_t, self.env.T)
                    candidates_for_shippers.append((score, dist, shipper, pickup_order))
            
            if not candidates_for_shippers:
                # Các shipper còn lại không tìm được đơn hàng nào phù hợp
                hotspots = self.detector.predicted_hotspots if (self.enable_reposition and hasattr(self, "detector") and self.detector.is_surge and self.detector.predicted_hotspots) else []
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
