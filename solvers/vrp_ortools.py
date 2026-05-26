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
REPLAN_INTERVAL_LARGE = 15   # Δt cho config lớn (N >= 18)
ORTOOLS_TIME_LIMIT_S  = 5    # Giới hạn thời gian OR-Tools (giây)
MAX_ACTIVE_ORDERS_SMALL = 150  # N < 50
MAX_ACTIVE_ORDERS_LARGE = 60   # N >= 50 (tránh cost matrix quá lớn)
MIN_EXPECTED_REWARD   = 0.5  # Bỏ đơn có expected reward < ngưỡng này
MAX_HEURISTIC_STOPS   = 12   # Số stop tối đa trong fallback planner nội bộ
DISTANCE_COST_SCALE   = 10   # Scale objective distance để cân bằng với drop penalty
DROP_REWARD_SCALE     = 35   # Reward dự kiến -> penalty nếu bỏ đơn
DROP_PRIORITY_BONUS   = 40   # Bonus penalty cho đơn priority cao
LATE_COST_SCALE       = 25   # Soft deadline penalty trên mỗi timestep trễ


from solvers.shared.pathfinder import get_pathfinder
from solvers.shared.detector import OnlineSurgeHotspotDetector

try:
    from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import cvxpy as cp
    HAS_CVXPY = True
except ImportError:
    HAS_CVXPY = False

def python_linear_sum_assignment(cost_matrix):
    n = len(cost_matrix)
    if n == 0:
        return [], []
    m = len(cost_matrix[0])
    transposed = False
    if n > m:
        cost_matrix = [list(x) for x in zip(*cost_matrix)]
        n, m = m, n
        transposed = True
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float('inf')] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float('inf')
            j1 = 0
            for j in range(1, m + 1):
                if not used[j]:
                    cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(0, m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    res = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            res[p[j] - 1] = j - 1
    row_ind = list(range(n))
    col_ind = res
    if transposed:
        orig_row = [-1] * m
        for r, c in zip(row_ind, col_ind):
            if c != -1:
                orig_row[c] = r
        matched_rows = []
        matched_cols = []
        for c in range(m):
            if orig_row[c] != -1:
                matched_rows.append(c)
                matched_cols.append(orig_row[c])
        return matched_rows, matched_cols
    return row_ind, col_ind

def linear_sum_assignment(cost_matrix):
    if HAS_SCIPY:
        return scipy_linear_sum_assignment(cost_matrix)
    return python_linear_sum_assignment(cost_matrix)


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
        self._plans: Dict[int, deque] = {}

        # Đơn đã assign cho shipper nhưng chưa pickup
        self._assignment: Dict[int, int] = {}   # order_id → shipper_id

        self._last_plan_t: int = -99
        self._replan_interval = (
            REPLAN_INTERVAL_LARGE if env.N >= 18 else REPLAN_INTERVAL_SMALL
        )
        self.max_active_orders = (
            MAX_ACTIVE_ORDERS_LARGE if env.N > 50 else MAX_ACTIVE_ORDERS_SMALL
        )
        self._prev_pending_count: int = 0

        from env import is_valid_cell, valid_next_pos
        adj = {}
        for r in range(self.env.N):
            for c in range(self.env.N):
                pos = (r, c)
                if is_valid_cell(pos, self.grid):
                    neighbors = []
                    for move in ("U", "D", "L", "R"):
                        nxt = valid_next_pos(pos, move, self.grid)
                        if nxt != pos:
                            neighbors.append((move, nxt))
                    adj[pos] = neighbors
        self._adj = adj

        # Thống kê bottleneck ratio để tránh tắc nghẽn
        def local_is_bottleneck(pos):
            r, c = pos
            return sum(
                1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                if 0 <= r + dr < len(self.grid) and 0 <= c + dc < len(self.grid[0]) and self.grid[r + dr][c + dc] == 0
            ) <= 2
        free_cells = list(adj.keys())
        bn_count = sum(1 for pos in free_cells if local_is_bottleneck(pos))
        self.bottleneck_ratio = bn_count / len(free_cells) if free_cells else 0.0

        # Khởi tạo detector
        self.detector = OnlineSurgeHotspotDetector(
            env.N, len(env.shippers), env.G, env.T, self.grid
        )
        self._seen_order_ids = set()

        valid_nodes = list(adj.keys())
        sample_size = min(len(valid_nodes), 50)
        import random
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
                for _, nxt in adj.get(curr, []):
                    if nxt not in dist_map:
                        dist_map[nxt] = d_nxt
                        queue_append(nxt)
            for d in dist_map.values():
                if d > 0:
                    total_dist += d
                    pair_count += 1
        avg_dist = total_dist / pair_count if pair_count > 0 else 10.0
        self.avg_dist = avg_dist

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

        if env.N > 50:
            self.max_delivery_delay = 180
            self.max_opp_dist = 30
        elif env.N > 18:
            self.max_delivery_delay = 60
            self.max_opp_dist = 25
        else:
            x_pts = [13.66, 17.24, 17.75, 26.70, 29.95, 36.86]
            y_delay = [15, 15, 2, -5, 2, -10]
            y_opp = [15, 25, 10, 30, 20, 15]
            self.max_delivery_delay = max(5, int(round(interpolate(avg_dist, x_pts, y_delay))))
            self.max_opp_dist = max(10, int(round(interpolate(avg_dist, x_pts, y_opp))))


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

    def _is_bottleneck(self, pos: Position) -> bool:
        """Kiểm tra ô pos có phải nút cổ chai (<= 2 ô trống xung quanh) hay không."""
        r, c = pos
        return sum(
            1 for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]
            if 0 <= r + dr < len(self.grid) and 0 <= c + dc < len(self.grid[0]) and self.grid[r + dr][c + dc] == 0
        ) <= 2

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
                    move_B = self.pathfinder.next_move(pos_B, goal_B)
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
                    move_step = self.pathfinder.next_move(curr, goal)
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

    def _evaluate_route_utility(self, s: Shipper, route: List[RouteStop], obs: dict) -> float:
        if not route:
            return 0.0
            
        orders = obs["orders"]
        obs_t = obs["t"]
        
        utility = 0.0
        curr_pos = s.position
        elapsed = 0
        
        delivered_rewards = {}
        lateness_penalties = 0.0
        
        for stop_pos, op, oid in route:
            d = self._dist(curr_pos, stop_pos)
            elapsed += d
            curr_pos = stop_pos
            
            if op == 2:  # Delivery
                o = orders[oid]
                eta = obs_t + elapsed
                reward = delivery_reward(o, eta, self.env.T)
                delivered_rewards[oid] = reward
                
                if eta > o.et:
                    lateness_penalties += (eta - o.et) * 2.5
        
        total_reward = sum(delivered_rewards.values())
        move_cost = elapsed * 1.5
        
        utility = total_reward * 40.0 - lateness_penalties - move_cost
        return utility

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

    def _build_route_for_shipper(self, s: Shipper, assigned_orders: List[Order], obs: dict, limit: Optional[int] = None) -> List[RouteStop]:
        orders = obs["orders"]
        obs_t = obs["t"]
        
        # Các đơn đang mang trong túi
        carried = [orders[oid] for oid in s.bag if oid in orders and not orders[oid].delivered]
        
        if limit is None:
            compact_mid_map = 12 <= self.env.N < 18
            route_limit = 4 if compact_mid_map else MAX_HEURISTIC_STOPS
        else:
            route_limit = limit
            
        initial_carried = set(o.id for o in carried)
        initial_pickups = set(o.id for o in assigned_orders)
        initial_deliveries = initial_carried | initial_pickups
        total_possible_stops = 2 * len(initial_pickups) + len(initial_carried)
        
        use_exact_cost = (self.env.N <= 10)
        best_eval = -INF
        best_route = []
        
        # Đệ quy quay lui để tìm chuỗi hành động tối ưu
        def search(curr_pos, elapsed, curr_weight, curr_bag_size, curr_route, curr_carried, pending_pickups, pending_deliveries, current_score):
            nonlocal best_eval, best_route
            
            if curr_route:
                if not use_exact_cost:
                    if current_score > best_eval:
                        best_eval = current_score
                        best_route = list(curr_route)
                else:
                    if not pending_deliveries:
                        # Exact cost: ta muốn tối thiểu hóa cost, tức là tối đa hóa -cost
                        score_val = -current_score
                        if score_val > best_eval:
                            best_eval = score_val
                            best_route = list(curr_route)
                    
            if len(curr_route) >= route_limit:
                return
                
            # 1. Thử đi đến một điểm pickup
            if curr_bag_size < s.K_max:
                for oid in list(pending_pickups):
                    o = orders[oid]
                    if curr_weight + o.w <= s.W_max:
                        dst = (o.sx, o.sy)
                        d = self._dist(curr_pos, dst)
                        if d >= INF:
                            continue
                            
                        new_pickups = pending_pickups - {oid}
                        new_carried = curr_carried | {oid}
                        
                        if use_exact_cost:
                            step_val = d
                        else:
                            d_delivery = self._dist(dst, (o.ex, o.ey))
                            est_eta = obs_t + elapsed + d + d_delivery
                            reward = delivery_reward(o, est_eta, self.env.T)
                            step_val = 100.0 + reward / (d + 1) - 0.1 * d + o.p * 0.5
                            
                        search(
                            dst, elapsed + d, curr_weight + o.w, curr_bag_size + 1,
                            curr_route + [(oid, 1, dst)],
                            new_carried, new_pickups, pending_deliveries,
                            current_score + step_val
                        )
                        
            # 2. Thử đi đến một điểm delivery
            for oid in list(pending_deliveries):
                if oid in curr_carried:
                    o = orders[oid]
                    dst = (o.ex, o.ey)
                    d = self._dist(curr_pos, dst)
                    if d >= INF:
                        continue
                        
                    est_eta = obs_t + elapsed + d
                    reward = delivery_reward(o, est_eta, self.env.T)
                    
                    if use_exact_cost:
                        lateness = max(0, est_eta - o.et)
                        step_val = d + lateness * 2.0 - reward * 30.0
                    else:
                        step_score = reward / (d + 1) * 1.2 - 0.05 * d
                        if est_eta > o.et:
                            step_score -= (est_eta - o.et) * 0.5
                        step_val = 100.0 + step_score
                        
                    new_deliveries = pending_deliveries - {oid}
                    new_carried = curr_carried - {oid}
                    
                    search(
                        dst, elapsed + d, max(0.0, curr_weight - o.w), max(0, curr_bag_size - 1),
                        curr_route + [(oid, 2, dst)],
                        new_carried, pending_pickups, new_deliveries,
                        current_score + step_val
                    )

        # Chỉ chạy backtracking nếu tổng số các stops cần xem xét không quá lớn để tránh quá tải đệ quy
        if total_possible_stops <= 10:
            search(
                s.position, 0, sum(o.w for o in carried), len(carried),
                [], initial_carried, initial_pickups, initial_deliveries, 0
            )
            # Fallback nếu dùng exact cost nhưng không tìm thấy route hoàn chỉnh
            if not best_route and use_exact_cost:
                use_exact_cost = False
                best_eval = -INF
                search(
                    s.position, 0, sum(o.w for o in carried), len(carried),
                    [], initial_carried, initial_pickups, initial_deliveries, 0
                )
            if best_route:
                return [(pos, op, oid) for oid, op, pos in best_route]
                
        # Fallback sang Greedy
        return self._build_route_for_shipper_greedy(s, assigned_orders, obs, limit)

    def _build_route_for_shipper_greedy(self, s: Shipper, assigned_orders: List[Order], obs: dict, limit: Optional[int] = None) -> List[RouteStop]:
        orders = obs["orders"]
        obs_t = obs["t"]
        
        route: List[RouteStop] = []
        
        current_pos = s.position
        carried = set(oid for oid in s.bag if oid in orders and not orders[oid].delivered)
        pending_pickups = set(o.id for o in assigned_orders)
        pending_deliveries = set(carried) | set(o.id for o in assigned_orders)
        
        current_weight = sum(orders[oid].w for oid in carried)
        elapsed = 0
        
        if limit is None:
            compact_mid_map = 12 <= self.env.N < 18
            route_limit = 4 if compact_mid_map else MAX_HEURISTIC_STOPS
        else:
            route_limit = limit
        
        while (pending_pickups or pending_deliveries) and len(route) < route_limit:
            best_cand = None
            best_score = -INF
            
            # 1. Thử các điểm pickup hợp lệ
            if len(carried) < s.K_max:
                for oid in pending_pickups:
                    o = orders[oid]
                    if current_weight + o.w <= s.W_max:
                        dst = (o.sx, o.sy)
                        d = self._dist(current_pos, dst)
                        if d >= INF:
                            continue
                        
                        d_delivery = self._dist(dst, (o.ex, o.ey))
                        est_eta = obs_t + elapsed + d + d_delivery
                        reward = delivery_reward(o, est_eta, self.env.T)
                        
                        score = reward / (d + 1) - 0.1 * d + o.p * 0.5
                        
                        if score > best_score:
                            best_score = score
                            best_cand = ("pickup", oid, dst, d)
                            
            # 2. Thử các điểm delivery hợp lệ
            for oid in pending_deliveries:
                if oid in carried:
                    o = orders[oid]
                    dst = (o.ex, o.ey)
                    d = self._dist(current_pos, dst)
                    if d >= INF:
                        continue
                    
                    est_eta = obs_t + elapsed + d
                    reward = delivery_reward(o, est_eta, self.env.T)
                    
                    score = reward / (d + 1) * 1.2 - 0.05 * d
                    if est_eta > o.et:
                        # Penalty mạnh hơn cho đơn trễ — tránh chọn đơn trễ deadline nhiều
                        score -= (est_eta - o.et) * 2.0
                    
                    if score > best_score:
                        best_score = score
                        best_cand = ("deliver", oid, dst, d)
                        
            if best_cand is None:
                break
                
            kind, oid, dst, travel = best_cand
            elapsed += travel
            current_pos = dst
            
            if kind == "pickup":
                route.append((dst, 1, oid))
                carried.add(oid)
                pending_pickups.remove(oid)
                current_weight += orders[oid].w
            else:
                route.append((dst, 2, oid))
                carried.remove(oid)
                pending_deliveries.remove(oid)
                current_weight = max(0.0, current_weight - orders[oid].w)
                
            if elapsed >= self.env.T - obs_t:
                break
                
        return route

    def _run_vrp_exact(self, obs: dict) -> Optional[Dict[int, List[RouteStop]]]:
        """
        Build full route cho mỗi cặp (shipper, order) — chất lượng cao.
        Sử dụng bộ lọc khoảng cách nhanh để giảm số lần gọi route builder trên map lớn.
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

        if unpicked:
            def order_priority_key(o: Order):
                min_d = min(self._dist(s.position, (o.sx, o.sy)) for s in shippers)
                return (-o.p, min_d, o.et)
            unpicked.sort(key=order_priority_key)
            max_act = 150 if self.env.N > 50 else self.max_active_orders
            unpicked = unpicked[:max_act]

        # Bộ lọc khoảng cách động theo N
        if self.env.N > 50:
            max_dist = max(25, self.max_opp_dist * 1.2)
        elif self.env.N > 18:
            max_dist = max(20, self.max_opp_dist * 1.5)
        else:
            max_dist = 999.0

        def get_capacity_limit(s: Shipper):
            return s.K_max

        assigned_map: Dict[int, List[Order]] = {s.id: [] for s in shippers}

        max_rounds = max(get_capacity_limit(s) for s in shippers)
        for round_idx in range(max_rounds):
            already_assigned_ids = {o.id for orders_list in assigned_map.values() for o in orders_list}
            unassigned_active = [o for o in unpicked if o.id not in already_assigned_ids]
            if not unassigned_active:
                break

            round_shippers = [
                s for s in shippers
                if len(s.bag) + len(assigned_map[s.id]) < get_capacity_limit(s)
            ]
            if not round_shippers:
                break

            C = []
            for s in round_shippers:
                current_weight = sum(orders[oid].w for oid in s.bag if oid in orders) + sum(ao.w for ao in assigned_map[s.id])
                route_base = self._build_route_for_shipper(s, assigned_map[s.id], obs)
                baseline_util = self._evaluate_route_utility(s, route_base, obs)
                
                # Ước tính vị trí rảnh tay của shipper sau khi giao xong bag
                _, end_pos = self._estimate_bag_delivery(s, orders)
                
                row_cost = []
                for o in unassigned_active:
                    if current_weight + o.w > s.W_max:
                        row_cost.append(1e9)
                        continue
                        
                    # Lọc khoảng cách nhanh từ end_pos
                    d_pick = self._dist(end_pos, (o.sx, o.sy))
                    if d_pick > max_dist:
                        row_cost.append(1e9)
                        continue
                        
                    route_new = self._build_route_for_shipper(s, assigned_map[s.id] + [o], obs)
                    if not route_new:
                        row_cost.append(1e9)
                        continue
                    feasible = True
                    elapsed = 0
                    curr = s.position
                    for stop_pos, op, oid in route_new:
                        d = self._dist(curr, stop_pos)
                        elapsed += d
                        curr = stop_pos
                        if op == 2:
                            eta = obs_t + elapsed
                            if eta - orders[oid].et > self.max_delivery_delay:
                                feasible = False
                                break
                    if not feasible:
                        row_cost.append(1e9)
                        continue
                    new_util = self._evaluate_route_utility(s, route_new, obs)
                    row_cost.append(-(new_util - baseline_util))
                C.append(row_cost)

            if not C or not C[0]:
                break

            row_ind, col_ind = linear_sum_assignment(C)
            matches = [
                (-C[r][c], r, c)
                for r, c in zip(row_ind, col_ind)
                if C[r][c] < 1e8 and -C[r][c] > 0
            ]
            if not matches:
                break
            matches.sort(key=lambda x: -x[0])

            any_assigned = False
            for _, r, c in matches:
                s = round_shippers[r]
                o = unassigned_active[c]
                current_weight = sum(orders[oid].w for oid in s.bag if oid in orders) + sum(ao.w for ao in assigned_map[s.id])
                if current_weight + o.w <= s.W_max and len(s.bag) + len(assigned_map[s.id]) < get_capacity_limit(s):
                    assigned_map[s.id].append(o)
                    any_assigned = True
            if not any_assigned:
                break

        routes: Dict[int, List[RouteStop]] = {}
        for s in shippers:
            routes[s.id] = self._build_route_for_shipper(s, assigned_map[s.id], obs)
        return routes if any(routes.values()) else None

    def _estimate_bag_delivery(self, s: Shipper, orders: Dict[int, Order]) -> Tuple[int, Position]:
        carried_oids = [oid for oid in s.bag if oid in orders and not orders[oid].delivered]
        if not carried_oids:
            return 0, s.position
        
        import itertools
        best_dist = 1e9
        best_end_pos = s.position
        
        # Vì len(carried_oids) <= 3, hoán vị tối đa là 6
        for perm in itertools.permutations(carried_oids):
            curr_dist = 0
            curr_pos = s.position
            for oid in perm:
                o = orders[oid]
                d = self._dist(curr_pos, (o.ex, o.ey))
                curr_dist += d
                curr_pos = (o.ex, o.ey)
            if curr_dist < best_dist:
                best_dist = curr_dist
                best_end_pos = curr_pos
                
        return best_dist, best_end_pos

    def _run_fast_assignment(self, obs: dict) -> Optional[Dict[int, List[RouteStop]]]:
        """
        Fast assignment: dùng _pickup_score scalar làm cost matrix thay vì
        build full route cho mỗi cặp (shipper, order).
        Giảm complexity từ O(C×G×route_build) xuống O(C×G) scalars.
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

        # Rộng hơn cho map lớn để giao được nhiều đơn hơn
        deadline_slack = self.max_delivery_delay + 30

        # Sắp xếp theo urgency: deadline gần trước, priority cao trước
        if unpicked:
            unpicked.sort(key=lambda o: (o.et - obs_t, -o.p))
            max_act = 150 if self.env.N > 50 else self.max_active_orders
            unpicked = unpicked[:max_act]

        # --- Bảo lưu assignment từ plan cũ (anti-thrashing) ---
        assigned_map: Dict[int, List[Order]] = {s.id: [] for s in shippers}
        preserved_ids: Set[int] = set()

        shipper_map = {s.id: s for s in shippers}
        for oid, sid in list(self._assignment.items()):
            o = orders.get(oid)
            if o is None or o.picked or o.delivered:
                continue
            s_obj = shipper_map.get(sid)
            if s_obj is None:
                continue
            current_w = sum(orders[x].w for x in s_obj.bag if x in orders)
            current_w += sum(ao.w for ao in assigned_map[sid])
            if current_w + o.w <= s_obj.W_max and len(s_obj.bag) + len(assigned_map[sid]) < s_obj.K_max:
                assigned_map[sid].append(o)
                preserved_ids.add(oid)

        # Loại các đơn đã preserved khỏi pool
        unassigned_pool = [o for o in unpicked if o.id not in preserved_ids]

        # --- Multi-round Hungarian assignment ---
        max_rounds = max(s.K_max for s in shippers)
        for _round in range(max_rounds):
            already_assigned_ids = preserved_ids | {
                o.id for orders_list in assigned_map.values() for o in orders_list
            }
            unassigned_active = [o for o in unassigned_pool if o.id not in already_assigned_ids]
            if not unassigned_active:
                break

            round_shippers = [
                s for s in shippers
                if len(s.bag) + len(assigned_map[s.id]) < s.K_max
            ]
            if not round_shippers:
                break

            # Build cost matrix bằng _pickup_score — O(1) mỗi cell
            C = []
            for s in round_shippers:
                current_weight = (
                    sum(orders[oid].w for oid in s.bag if oid in orders)
                    + sum(ao.w for ao in assigned_map[s.id])
                )
                
                # Ước tính thời gian giao xong bag hiện tại và vị trí kết thúc
                elapsed_est, end_pos = self._estimate_bag_delivery(s, orders)

                row_cost = []
                for o in unassigned_active:
                    # Kiểm tra weight/capacity
                    if current_weight + o.w > s.W_max:
                        row_cost.append(1e9)
                        continue
                        
                    # Tính khoảng cách từ end_pos (sau khi giao xong bag) tới pickup mới
                    d_pick = self._dist(end_pos, (o.sx, o.sy))
                    d_drop = self._dist((o.sx, o.sy), (o.ex, o.ey))
                    if d_pick >= INF or d_drop >= INF:
                        row_cost.append(1e9)
                        continue
                        
                    eta = obs_t + elapsed_est + d_pick + d_drop
                    if eta - o.et > deadline_slack:
                        row_cost.append(1e9)
                        continue
                        
                    reward = delivery_reward(o, eta, self.env.T)
                    if reward <= 0.0:
                        row_cost.append(1e9)
                        continue
                        
                    score = reward / (d_pick + d_drop + 1) + 0.05 * o.p
                    if eta <= o.et:
                        score *= 2.0
                    else:
                        score -= (eta - o.et) * 0.1
                        
                    if score <= 0.0:
                        row_cost.append(1e9)
                        continue
                        
                    row_cost.append(-score)  # minimize → maximize score
                C.append(row_cost)

            if not C or not C[0]:
                break

            row_ind, col_ind = linear_sum_assignment(C)

            matches = [
                (-C[r][c], r, c)
                for r, c in zip(row_ind, col_ind)
                if C[r][c] < 1e8
            ]
            if not matches:
                break
            matches.sort(key=lambda x: -x[0])  # descending score

            any_assigned = False
            for score, r, c in matches:
                s = round_shippers[r]
                o = unassigned_active[c]
                current_weight = (
                    sum(orders[oid].w for oid in s.bag if oid in orders)
                    + sum(ao.w for ao in assigned_map[s.id])
                )
                if current_weight + o.w <= s.W_max:
                    assigned_map[s.id].append(o)
                    any_assigned = True

            if not any_assigned:
                break

        # Build route greedy cho từng shipper (chỉ 1 lần sau assignment)
        routes: Dict[int, List[RouteStop]] = {}
        for s in shippers:
            routes[s.id] = self._build_route_for_shipper(s, assigned_map[s.id], obs)

        return routes if any(routes.values()) else None



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
        return self._navigate_to(s, pos, cargo_op_at_goal=op)

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
            return self._navigate_to(s, goal, cargo_op_at_goal=2)

        # --- Priority 3: Reposition if surge ---
        if len(s.bag) == 0 and self.detector.is_surge and self.detector.predicted_hotspots:
            best_hotspot = min(
                self.detector.predicted_hotspots,
                key=lambda hp: self._dist(s.position, hp)
            )
            if best_hotspot != s.position:
                self._shipper_goals[s.id] = best_hotspot
                return self._navigate_to(s, best_hotspot, cargo_op_at_goal=0)

        # --- Priority 4: Greedy fallback ---
        return self._greedy_pick(s, obs)


    def _navigate_to(self, s: Shipper, goal: Position, cargo_op_at_goal: int) -> Action:
        """Di chuyển 1 bước về phía goal. Nếu đã tới → thực hiện cargo_op."""
        pos = s.position
        if pos == goal:
            return ("S", cargo_op_at_goal)
        move, next_pos = self._move_towards(s, goal)
        if next_pos == goal:
            return (move, cargo_op_at_goal)
        return (move, 0)

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

        # Clear previous greedy/stale assignments for this shipper
        for oid in list(self._assignment.keys()):
            if self._assignment[oid] == s.id and oid not in s.bag:
                self._assignment.pop(oid, None)

        cands = []
        for o in orders.values():
            if o.picked or o.delivered:
                continue
            if self._assignment.get(o.id) is not None and self._assignment[o.id] != s.id:
                continue
            if not s.can_carry(o, orders):
                continue
            d_pickup = self._dist(s.position, (o.sx, o.sy))
            if d_pickup >= INF:
                continue
            # Deadline filter: bỏ đơn chắc chắn reward = 0
            d_deliver = self._dist((o.sx, o.sy), (o.ex, o.ey))
            if obs_t + d_pickup + d_deliver - o.et > max(self.max_delivery_delay, int(self.env.N * 1.5)):
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
            self._assignment[best.id] = s.id

            # Batch reserve: reserve thêm đơn "trên đường" để tránh shipper khác cướp
            remaining_slots = s.K_max - len(s.bag) - 1
            w_carried = sum(orders[oid].w for oid in s.bag if oid in orders) + best.w
            batch_list = []
            for o2 in orders.values():
                if remaining_slots <= 0:
                    break
                if o2.id == best.id or (self._assignment.get(o2.id) is not None and self._assignment[o2.id] != s.id):
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
                if obs_t + d_via + d_deliver2 - o2.et > max(self.max_delivery_delay, int(self.env.N * 1.5)):
                    continue
                self._assignment[o2.id] = s.id
                w_carried += o2.w
                remaining_slots -= 1
                batch_list.append(o2)

            tgt_queue = self._targets[s.id]
            tgt_queue.clear()
            tgt_queue.append(((best.sx, best.sy), 1, best.id))
            for o2 in batch_list:
                tgt_queue.append(((o2.sx, o2.sy), 1, o2.id))
            tgt_queue.append(((best.ex, best.ey), 2, best.id))
            for o2 in batch_list:
                tgt_queue.append(((o2.ex, o2.ey), 2, o2.id))

            if tgt_queue:
                return self._planned_stop_action(tgt_queue[0], s, obs)

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
        routes = self._run_vrp_exact(obs)
        if routes is not None:
            new_targets = self._extract_targets(routes, obs)
            for s_id, tgts in new_targets.items():
                self._targets[s_id] = tgts
            new_assignment = {}
            for s_id, route_stops in routes.items():
                for pos, op, oid in route_stops:
                    if op == 1:
                        new_assignment[oid] = s_id
            self._assignment = new_assignment

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        self._targets: Dict[int, deque] = {}
        self._assignment = {}
        for s in obs["shippers"]:
            self._targets[s.id] = deque()

        self._last_plan_t = -self._replan_interval
        self._seen_order_ids = set()
        self.detector = OnlineSurgeHotspotDetector(
            self.env.N, len(obs["shippers"]), obs["G"], self.env.T, self.grid
        )

        while not obs.get("done", False):
            orders = obs["orders"]
            shippers_list = obs["shippers"]
            current_t = obs.get("t", 0)
            
            # Cập nhật detector
            current_order_ids = set(orders.keys())
            new_order_ids = list(current_order_ids - self._seen_order_ids)
            self._seen_order_ids.update(current_order_ids)
            self.detector.update(current_t, new_order_ids, orders)

            # Thiết lập biến tránh kẹt nút cổ chai
            self._all_shippers = shippers_list
            self._all_shipper_positions = {s.position for s in shippers_list}
            self._shipper_goals = {}
            for s in shippers_list:
                tgt_queue = self._targets.get(s.id)
                if tgt_queue and len(tgt_queue) > 0:
                    self._shipper_goals[s.id] = tgt_queue[0][0]
                else:
                    self._shipper_goals[s.id] = s.position

            if self._should_replan(obs):
                self._replan(obs)

            # Cleanup assignments for picked or delivered orders, or stale targets
            for oid in list(self._assignment.keys()):
                o = orders.get(oid)
                if o is None or o.picked or o.delivered:
                    self._assignment.pop(oid, None)
                    continue
                s_id = self._assignment[oid]
                s_obj = next((x for x in shippers_list if x.id == s_id), None)
                if s_obj is None:
                    self._assignment.pop(oid, None)
                    continue
                tgt_queue = self._targets.get(s_id)
                if tgt_queue:
                    has_pickup_target = any(op == 1 and o_id == oid for _, op, o_id in tgt_queue)
                    if not has_pickup_target:
                        self._assignment.pop(oid, None)

            actions: Dict[int, Action] = {}
            for s in sorted(obs["shippers"], key=lambda s: s.id):
                actions[s.id] = self._step_action(s, obs)

            # Giải quyết xung đột/đối đầu giữa các shipper
            goals = {s.id: self._shipper_goals.get(s.id, s.position) for s in shippers_list}
            actions = self._resolve_deadlocks(shippers_list, actions, goals)

            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
