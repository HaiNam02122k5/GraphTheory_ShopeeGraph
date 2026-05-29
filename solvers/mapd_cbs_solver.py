from __future__ import annotations
import random
import time
import heapq
from collections import deque
from typing import Deque, Dict, List, Tuple, Set, Optional, Any

from env import DeliveryEnv, Order, Shipper, is_valid_cell, valid_next_pos, delivery_reward
from solvers.shared.detector import OnlineSurgeHotspotDetector

Position = Tuple[int, int]
Move = str
Action = Tuple[Move, int]
RouteStop = Tuple[Position, int, int]  # (target position, cargo op, order id)
EdgeKey = Tuple[int, int, int]

INF = 10**9
REPLAN_INTERVAL_SMALL = 10
REPLAN_INTERVAL_LARGE = 20
MIN_EXPECTED_REWARD = 0.15

ACO_ALPHA = 1.0
ACO_BETA = 2.4
ACO_RHO = 0.35
ACO_Q = 3.0
TAU0 = 1.0
TAU_MIN = 0.05
TAU_MAX = 8.0

MOVES_CBS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1), "S": (0, 0)}
MOVE_TO_STR = {(-1, 0): "U", (1, 0): "D", (0, -1): "L", (0, 1): "R", (0, 0): "S"}

try:
    from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def python_linear_sum_assignment(cost_matrix: List[List[float]]) -> Tuple[List[int], List[int]]:
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
        minv = [float("inf")] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
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
            for j in range(m + 1):
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


def linear_sum_assignment(cost_matrix: List[List[float]]):
    if HAS_SCIPY:
        return scipy_linear_sum_assignment(cost_matrix)
    return python_linear_sum_assignment(cost_matrix)


class Solver:
    """Base class nhỏ gọn để MAPD-CBS tự chạy trong file này."""

    def __init__(self, env: DeliveryEnv):
        if not isinstance(env, DeliveryEnv):
            raise TypeError("Solver chi ho tro khoi tao dang Solver(env: DeliveryEnv).")
        self.env: DeliveryEnv = env
        self.grid = env.grid
        self.orders: List[Order] = []

    def run(self) -> dict:
        raise NotImplementedError


_PATHFINDER_CACHE: Dict[Tuple[Tuple[int, ...], ...], "PathFinder"] = {}


def get_pathfinder(grid: List[List[int]]) -> "PathFinder":
    grid_hash = tuple(tuple(row) for row in grid)
    if grid_hash not in _PATHFINDER_CACHE:
        _PATHFINDER_CACHE[grid_hash] = PathFinder(grid)
    return _PATHFINDER_CACHE[grid_hash]


class PathFinder:
    """
    Cached grid shortest-path helper. It uses the same optional PyTorch APSP
    acceleration as the standalone pathfinder, with a CPU BFS fallback.
    """

    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.N = len(grid)
        self._bfs_cache: Dict[Tuple[Position, Position], Tuple[int, List[Move]]] = {}
        self._distance_cache: Dict[Tuple[Position, Position], int] = {}
        self._next_move_cache: Dict[Tuple[Position, Position], Move] = {}
        self.has_torch = self._precompute_with_torch(grid)

    def _precompute_with_torch(self, grid: List[List[int]]) -> bool:
        try:
            import torch
        except ImportError:
            return False

        device = "cuda" if torch.cuda.is_available() else "cpu"
        N = self.N
        free_cells = [(r, c) for r in range(N) for c in range(N) if grid[r][c] == 0]
        V = len(free_cells)
        self.cell_to_idx = {cell: i for i, cell in enumerate(free_cells)}
        self.idx_to_cell = free_cells

        adj = torch.full((V + 1, 4), V, dtype=torch.long, device=device)
        moves_offset = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for idx, (r, c) in enumerate(free_cells):
            for dir_idx, (dr, dc) in enumerate(moves_offset):
                nr, nc = r + dr, c + dc
                if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] == 0:
                    adj[idx, dir_idx] = self.cell_to_idx[(nr, nc)]

        dist = torch.full((V + 1, V + 1), 9999, dtype=torch.int16, device=device)
        first_move = torch.full((V + 1, V + 1), 4, dtype=torch.int8, device=device)
        diag_indices = torch.arange(V + 1, device=device)
        dist[diag_indices, diag_indices] = 0

        visited = torch.zeros((V + 1, V + 1), dtype=torch.bool, device=device)
        visited[diag_indices, diag_indices] = True
        frontier = torch.zeros((V + 1, V + 1), dtype=torch.bool, device=device)

        for c in range(4):
            neighbors = adj[:, c]
            valid = neighbors < V
            if valid.any():
                src_indices = torch.arange(V + 1, device=device)[valid]
                dest_indices = neighbors[valid]
                dist[src_indices, dest_indices] = 1
                first_move[src_indices, dest_indices] = c
                frontier[src_indices, dest_indices] = True
                visited[src_indices, dest_indices] = True

        for d in range(2, min(V, N * 6)):
            new_frontier = torch.zeros((V + 1, V + 1), dtype=torch.bool, device=device)
            already_reached = torch.zeros((V + 1, V + 1), dtype=torch.bool, device=device)
            for c in range(4):
                reached = (frontier[:, adj[:, c]] & ~visited) & ~already_reached
                if reached.any():
                    first_move = torch.where(reached, first_move[:, adj[:, c]], first_move)
                    dist = torch.where(reached, torch.tensor(d, dtype=torch.int16, device=device), dist)
                    new_frontier |= reached
                    already_reached |= reached
            if not new_frontier.any():
                break
            visited |= new_frontier
            frontier = new_frontier

        self.dist_matrix = dist[:V, :V].cpu().numpy()
        self.next_move_matrix = first_move[:V, :V].cpu().numpy()
        self.adj_cpu = adj[:V].cpu().numpy()
        return True

    def _fallback_bfs(self, start: Position, goal: Position) -> Tuple[int, List[Move]]:
        if start == goal:
            return 0, []
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return INF, []

        parent: Dict[Position, Tuple[Optional[Position], Move]] = {start: (None, "S")}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            if cur == goal:
                break
            for mv in ("U", "D", "L", "R"):
                nxt = valid_next_pos(cur, mv, self.grid)
                if nxt != cur and nxt not in parent:
                    parent[nxt] = (cur, mv)
                    queue.append(nxt)

        if goal not in parent:
            return INF, []

        moves: List[Move] = []
        cur = goal
        while cur != start:
            prev, mv = parent[cur]
            moves.append(mv)
            cur = prev  # type: ignore[assignment]
        moves.reverse()
        return len(moves), moves

    def dist(self, start: Position, goal: Position) -> int:
        if self.has_torch:
            if start == goal:
                return 0
            s_idx = self.cell_to_idx.get(start)
            g_idx = self.cell_to_idx.get(goal)
            if s_idx is None or g_idx is None:
                return INF
            d = self.dist_matrix[s_idx, g_idx]
            return INF if d >= 9999 else int(d)

        key = (start, goal)
        if key not in self._distance_cache:
            d, _ = self._fallback_bfs(start, goal)
            self._distance_cache[key] = d
        return self._distance_cache[key]

    def path(self, start: Position, goal: Position) -> List[Move]:
        if self.has_torch:
            if start == goal:
                return []
            s_idx = self.cell_to_idx.get(start)
            g_idx = self.cell_to_idx.get(goal)
            if s_idx is None or g_idx is None:
                return []

            path_moves: List[Move] = []
            curr_idx = s_idx
            moves_list = ["U", "D", "L", "R"]
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

        key = (start, goal)
        if key not in self._bfs_cache:
            d, p = self._fallback_bfs(start, goal)
            self._bfs_cache[key] = (d, p)
        return self._bfs_cache[key][1]


class ACOSolver(Solver):
    """
    Rolling-horizon Ant Colony Optimization base used by MAPD-CBS for task
    allocation. Kept in this file so MAPD-CBS is self-contained.
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

    def _bfs(self, start: Position, goal: Position) -> Tuple[int, List[Move]]:
        return self.pathfinder.dist(start, goal), self.pathfinder.path(start, goal)

    def _dist(self, a: Position, b: Position) -> int:
        return self.pathfinder.dist(a, b)

    def _path(self, a: Position, b: Position) -> List[Move]:
        return self.pathfinder.path(a, b)

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
                key=lambda p: self._dist(p, (o.sx, o.sy)),
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
            sid, stop, order, travel, _, _ = candidates[selected_idx]

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

    def _replan(self, obs: dict) -> None:
        self._last_plan_t = obs["t"]
        routes = self._run_aco(obs)
        if routes is None:
            return
        for sid, route in routes.items():
            self._targets[sid] = deque(route)

class MAPDCBSSolver(ACOSolver):
    """
    MAPD-CBS Solver: 
    - High-level: ACO for task allocation.
    - Low-level: Windowed Conflict-Based Search for MAPF.
    """

    method_name = "MAPD-CBS"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)
        self.window_size = 12
        self.cbs_max_nodes = 30
        self.cbs_time_limit = 0.15
        self._cbs_mode = "full"
        self._group_cbs_max_size = 6
        self._aco_iterations_override: Optional[int] = None
        self._max_active_override: Optional[int] = None
        self._route_limit_override: Optional[int] = None
        self._strict_deadline_mode = False
        self._stats = {
            "aco_calls": 0,
            "cbs_calls": 0,
            "cbs_success": 0,
            "cbs_failed": 0,
            "group_cbs_calls": 0,
            "local_steps": 0,
            "surge_steps": 0,
            "hotspot_assignments": 0,
        }
        self._order_delivery_dist: Dict[int, int] = {}
        self.detector = OnlineSurgeHotspotDetector(
            env.N, env.C, env.G, env.T, self.grid
        )
        self._seen_order_ids: Set[int] = set()
        self._shipper_hotspot_assignments: Dict[int, Position] = {}
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
        self._configure_runtime_budget()

    def _configure_runtime_budget(self) -> None:
        if self.env.G >= 1000 or self.env.C >= 20:
            self.window_size = 6
            self.cbs_max_nodes = 10
            self.cbs_time_limit = 0.04
            self._replan_interval = 18
            self._aco_iterations_override = 12
            self._max_active_override = 110
            self._route_limit_override = 7
            self._cbs_mode = "group"
            self._group_cbs_max_size = 6
            self._strict_deadline_mode = True
        elif self.env.N >= 100:
            self.window_size = 6
            self.cbs_max_nodes = 8
            self.cbs_time_limit = 0.025
            self._replan_interval = 50
            self._aco_iterations_override = 4
            self._max_active_override = 45
            self._route_limit_override = 5
            self._cbs_mode = "group"
            self._group_cbs_max_size = 5
        elif self.env.N >= 50:
            self.window_size = 8
            self.cbs_max_nodes = 12
            self.cbs_time_limit = 0.05
            self._replan_interval = 35
            self._aco_iterations_override = 6
            self._max_active_override = 55
            self._route_limit_override = 6
            self._cbs_mode = "group"
            self._group_cbs_max_size = 6

    def _n_iterations(self) -> int:
        if self._aco_iterations_override is not None:
            return self._aco_iterations_override
        return super()._n_iterations()

    def _route_limit(self) -> int:
        if self._route_limit_override is not None:
            return self._route_limit_override
        return super()._route_limit()

    def _pickup_candidate_limit(self) -> int:
        if self.env.G >= 1000 or self.env.C >= 20:
            return 60
        if self.env.N >= 100:
            return 18
        if self.env.N >= 50:
            return 24
        return 30

    def _batch_candidate_limit(self) -> int:
        if self.env.G >= 1000 or self.env.C >= 20:
            return 30
        if self.env.N >= 100:
            return 8
        if self.env.N >= 50:
            return 10
        return 15

    def _get_order_delivery_dist(self, order: Order) -> int:
        if order.id in self._order_delivery_dist:
            return self._order_delivery_dist[order.id]
        dist = self._dist((order.sx, order.sy), (order.ex, order.ey))
        self._order_delivery_dist[order.id] = dist
        return dist

    def _order_pickup_score(self, shipper: Shipper, order: Order, current_t: int, T: int) -> float:
        dist_pickup = self._dist(shipper.position, (order.sx, order.sy))
        dist_deliver = self._get_order_delivery_dist(order)
        
        if dist_pickup >= INF or dist_deliver >= INF:
            return -INF
        
        t_estimated_delivery = current_t + dist_pickup + dist_deliver
        expected_reward = delivery_reward(order, t_estimated_delivery, T)
        
        expiry_mult = 1.0
        total_steps = max(dist_pickup + dist_deliver, 1)
        
        if t_estimated_delivery > order.et:
            urgency = 0.0
        else:
            time_slack = max(order.et - current_t, 1)
            urgency = 1.0 / time_slack
        
        deadline_penalty = 0.0
        if self._strict_deadline_mode:
            lateness = t_estimated_delivery - order.et
            if lateness > self.max_delivery_delay:
                deadline_penalty = lateness * 0.25

        return (expected_reward * expiry_mult) / total_steps + urgency * 10.0 - deadline_penalty

    def _select_pickup_v1(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: Set[int],
        current_t: int,
    ) -> Optional[Order]:
        candidates: List[Order] = []

        for order in orders.values():
            if order.id in reserved_order_ids:
                continue
            if not shipper.can_carry(order, orders):
                continue
            candidates.append(order)

        if not candidates:
            return None

        if len(self.grid) > 20:
            candidates.sort(key=lambda o: abs(shipper.position[0] - o.sx) + abs(shipper.position[1] - o.sy))
            candidates = candidates[:self._pickup_candidate_limit()]

        valid_candidates = []
        for order in candidates:
            if self._dist(shipper.position, (order.sx, order.sy)) < INF:
                valid_candidates.append(order)

        if not valid_candidates:
            return None

        if self._strict_deadline_mode:
            feasible_candidates = []
            fallback_candidates = []
            for order in valid_candidates:
                d_pick = self._dist(shipper.position, (order.sx, order.sy))
                d_drop = self._get_order_delivery_dist(order)
                eta = current_t + d_pick + d_drop + 2
                score = self._order_pickup_score(shipper, order, current_t, self.env.T)
                if eta - order.et <= self.max_delivery_delay:
                    feasible_candidates.append((score, order))
                elif score > 0:
                    fallback_candidates.append((score, order))

            scored = feasible_candidates or fallback_candidates
            if not scored:
                return None
            return max(scored, key=lambda item: (item[0], -item[1].id))[1]

        return max(
            valid_candidates,
            key=lambda order: (
                self._order_pickup_score(shipper, order, current_t, self.env.T),
                -order.id,
            ),
        )

    def _estimate_bag_reward(
        self,
        start_pos: Position,
        bag_orders: List[Order],
        current_t: int,
        T: int,
    ) -> float:
        if not bag_orders:
            return 0.0

        total_reward = 0.0
        pos = start_pos
        t = current_t

        for order in sorted(bag_orders, key=lambda o: o.et):
            goal = (order.ex, order.ey)
            d = self._dist(pos, goal)
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
        pos = start_pos
        t = current_t
        for order in sorted(bag_orders, key=lambda o: o.et):
            d = self._dist(pos, (order.ex, order.ey))
            if d < INF:
                t += d
                pos = (order.ex, order.ey)
        return t

    def _estimate_last_position(
        self,
        start_pos: Position,
        bag_orders: List[Order],
    ) -> Position:
        pos = start_pos
        for order in sorted(bag_orders, key=lambda o: o.et):
            d = self._dist(pos, (order.ex, order.ey))
            if d < INF:
                pos = (order.ex, order.ey)
        return pos

    def _estimate_route_times(
        self,
        shipper_pos: Position,
        pickup_orders: List[Order],
        bag_orders: List[Order],
        start_t: int,
    ) -> Dict[int, int]:
        t = start_t
        curr = shipper_pos

        for order in pickup_orders:
            dist = self._dist(curr, (order.sx, order.sy))
            if dist >= INF:
                return {}
            t += dist + 1
            curr = (order.sx, order.sy)

        remaining = list(pickup_orders) + list(bag_orders)
        delivery_times: Dict[int, int] = {}
        while remaining:
            best = min(
                remaining,
                key=lambda o: (
                    0 if t + self._dist(curr, (o.ex, o.ey)) <= o.et else 1,
                    o.et,
                    self._dist(curr, (o.ex, o.ey)),
                    -o.p,
                    o.id,
                ),
            )
            dist = self._dist(curr, (best.ex, best.ey))
            if dist >= INF:
                return {}
            t += dist + 1
            delivery_times[best.id] = t
            curr = (best.ex, best.ey)
            remaining.remove(best)

        return delivery_times

    def _evaluate_opportunistic_pickup(
        self,
        shipper: Shipper,
        candidate: Order,
        current_t: int,
        orders: Dict[int, Order],
    ) -> float:
        T = self.env.T
        bag_orders = [
            orders[oid] for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]

        cand_pickup = (candidate.sx, candidate.sy)
        cand_delivery = (candidate.ex, candidate.ey)

        d_to_cpickup = self._dist(shipper.position, cand_pickup)
        d_cpickup_cdel = self._get_order_delivery_dist(candidate)

        if d_to_cpickup >= INF or d_cpickup_cdel >= INF:
            return -INF

        baseline_reward = self._estimate_bag_reward(
            shipper.position, bag_orders, current_t, T
        )

        pos_after_pickup = cand_pickup
        t_after_pickup = current_t + d_to_cpickup

        bag_reward_after_pickup = self._estimate_bag_reward(
            pos_after_pickup, bag_orders, t_after_pickup, T
        )

        t_finish_bag = self._estimate_finish_time(
            pos_after_pickup, bag_orders, t_after_pickup
        )
        last_bag_pos = self._estimate_last_position(pos_after_pickup, bag_orders)
        d_last_to_cdel = self._dist(last_bag_pos, cand_delivery)
        t_cand_delivery = t_finish_bag + d_last_to_cdel
        reward_cand = delivery_reward(candidate, t_cand_delivery, T)

        w_extra = candidate.w
        extra_move_cost = d_to_cpickup * (-0.01 * w_extra / max(shipper.W_max, 1.0))

        new_total = bag_reward_after_pickup + reward_cand + extra_move_cost
        net_gain = new_total - baseline_reward

        return net_gain

    def _find_opportunistic_pickup(
        self,
        shipper: Shipper,
        orders: Dict[int, Order],
        reserved_order_ids: Set[int],
        current_t: int,
    ) -> Optional[Order]:
        current_weight = sum(
            orders[oid].w for oid in shipper.bag if oid in orders
        )
        if (len(shipper.bag) >= shipper.K_max
                or current_weight >= shipper.W_max):
            return None

        best_order: Optional[Order] = None
        best_gain = 0.0

        for order in orders.values():
            if order.picked or order.delivered:
                continue
            if order.id in reserved_order_ids:
                continue
            if not shipper.can_carry(order, orders):
                continue
            if order.et <= current_t:
                continue

            gain = self._evaluate_opportunistic_pickup(
                shipper, order, current_t, orders
            )
            if gain > best_gain:
                best_gain = gain
                best_order = order

        return best_order

    def _plan_multi_pickup_route(self, shipper: Shipper, available_orders: Dict[int, Order], current_t: int) -> Optional[Order]:
        candidates = [
            o for o in available_orders.values()
            if not o.picked and not o.delivered
        ]
        if not candidates:
            return None
        
        if len(self.grid) > 20:
            candidates.sort(key=lambda o: abs(shipper.position[0] - o.sx) + abs(shipper.position[1] - o.sy))
            candidates = candidates[:self._pickup_candidate_limit()]
        
        candidates = [o for o in candidates if self._dist(shipper.position, (o.sx, o.sy)) < INF]
        if not candidates:
            return None
        
        route, total_weight, total_slots = [], 0.0, 0
        current_pos = shipper.position
        remaining = list(candidates)
        
        while remaining and total_slots < shipper.K_max:
            valid_rem = [o for o in remaining 
                         if total_weight + o.w <= shipper.W_max 
                         and total_slots + 1 <= shipper.K_max]
            if not valid_rem:
                break
                
            if len(self.grid) > 20:
                valid_rem.sort(key=lambda o: abs(current_pos[0] - o.sx) + abs(current_pos[1] - o.sy))
                valid_rem = valid_rem[:self._batch_candidate_limit()]
            
            found_next = False
            for best in sorted(
                valid_rem,
                key=lambda o: (
                    self._dist(current_pos, (o.sx, o.sy)),
                    o.et,
                    -o.p,
                    o.id,
                ),
            ):
                d_to_pickup = self._dist(current_pos, (best.sx, best.sy))
                d_to_delivery = self._get_order_delivery_dist(best)
                if d_to_pickup >= INF or d_to_delivery >= INF:
                    remaining.remove(best)
                    continue

                t_estimated = current_t + d_to_pickup + d_to_delivery + 2
                if self._strict_deadline_mode and t_estimated - best.et > self.max_delivery_delay:
                    remaining.remove(best)
                    continue
                if delivery_reward(best, t_estimated, self.env.T) <= 0:
                    remaining.remove(best)
                    continue

                test_route = route + [best]
                delivery_times = self._estimate_route_times(
                    shipper.position, test_route, [], current_t
                )
                if not delivery_times:
                    remaining.remove(best)
                    continue

                if self._strict_deadline_mode:
                    feasible = all(
                        delivery_times.get(order.id, INF) - order.et <= self.max_delivery_delay
                        for order in test_route
                    )
                    if not feasible:
                        remaining.remove(best)
                        continue

                route = test_route
                current_pos = (best.sx, best.sy)
                total_weight += best.w
                total_slots += 1
                remaining.remove(best)
                found_next = True
                break

            if not found_next:
                break
        
        return route[0] if route else None

    def _resolve_deadlocks(self, shippers: List[Shipper], actions: Dict[int, Action]) -> Dict[int, Action]:
        resolved = dict(actions)
        positions = {s.id: s.position for s in shippers}
        shipper_by_id = {s.id: s for s in shippers}
        
        desired = {}
        for s in shippers:
            move, op = resolved.get(s.id, ("S", 0))
            desired[s.id] = valid_next_pos(s.position, move, self.grid)

        def priority_key(sid: int) -> Tuple[int, int]:
            shipper = shipper_by_id[sid]
            _, op = resolved.get(sid, ("S", 0))
            if shipper.bag or op == 2:
                return (0, sid)
            if op == 1:
                return (1, sid)
            return (2, sid)

        def safe_yield_move(sid: int, avoid: Set[Position]) -> Optional[Move]:
            shipper = shipper_by_id[sid]
            occupied_now = set(positions.values())
            reserved_next = {pos for other, pos in desired.items() if other != sid}
            candidates: List[Tuple[int, Move]] = []
            for move in ("U", "D", "L", "R"):
                nxt = valid_next_pos(shipper.position, move, self.grid)
                if nxt == shipper.position:
                    continue
                if nxt in occupied_now or nxt in reserved_next:
                    continue
                candidates.append((1 if nxt in avoid else 0, move))
            if not candidates:
                return None
            candidates.sort()
            return candidates[0][1]

        # A loaded shipper should not wait behind an idle shipper if the idle
        # shipper can step aside locally.
        for mover in sorted(shippers, key=lambda s: priority_key(s.id)):
            mover_target = desired.get(mover.id, mover.position)
            if mover_target == mover.position:
                continue
            for blocker in shippers:
                if blocker.id == mover.id:
                    continue
                if positions[blocker.id] != mover_target:
                    continue
                blocker_action = resolved.get(blocker.id, ("S", 0))
                blocker_idle = (
                    not blocker.bag
                    and blocker_action[0] == "S"
                    and blocker_action[1] == 0
                    and desired.get(blocker.id) == blocker.position
                )
                mover_has_work = bool(mover.bag) or resolved.get(mover.id, ("S", 0))[1] in {1, 2}
                if not blocker_idle or not mover_has_work:
                    continue
                yield_move = safe_yield_move(blocker.id, {mover.position, mover_target})
                if yield_move is not None:
                    resolved[blocker.id] = (yield_move, 0)
                    desired[blocker.id] = valid_next_pos(blocker.position, yield_move, self.grid)
                break

        # Resolve same-cell next-step conflicts cheaply. The lower priority
        # shipper waits or sidesteps.
        target_to_sids: Dict[Position, List[int]] = {}
        for sid, pos in desired.items():
            target_to_sids.setdefault(pos, []).append(sid)
        for target, sids in target_to_sids.items():
            if len(sids) <= 1:
                continue
            ordered = sorted(sids, key=priority_key)
            for sid in ordered[1:]:
                yield_move = safe_yield_move(sid, {target})
                if yield_move is not None:
                    resolved[sid] = (yield_move, 0)
                    desired[sid] = valid_next_pos(positions[sid], yield_move, self.grid)
                else:
                    resolved[sid] = ("S", 0)
                    desired[sid] = positions[sid]

        for s1 in shippers:
            for s2 in shippers:
                if s1.id >= s2.id:
                    continue
                u, v = s1.id, s2.id
                pos_u, pos_v = positions[u], positions[v]
                des_u, des_v = desired[u], desired[v]
                
                if des_u == pos_v and des_v == pos_u and pos_u != pos_v:
                    evader = v
                    other = u
                    evader_pos = pos_v
                    other_pos = pos_u
                    
                    moved = False
                    for m in ("U", "D", "L", "R"):
                        nxt = valid_next_pos(evader_pos, m, self.grid)
                        if nxt != evader_pos and nxt != other_pos and nxt not in positions.values():
                            resolved[evader] = (m, 0)
                            desired[evader] = nxt
                            moved = True
                            break
                            
                    if not moved:
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
                        resolved[evader] = ("S", 0)
                        desired[evader] = evader_pos
                        
        return resolved

    def _drop_stale_order_targets(self, sid: int, oid: int) -> None:
        queue = self._targets.get(sid)
        if not queue:
            return
        self._targets[sid] = deque(stop for stop in queue if stop[2] != oid)

    def _is_stop_actionable(self, stop: Tuple[Position, int, int], shipper: Shipper, obs: dict) -> bool:
        if not super()._is_stop_actionable(stop, shipper, obs):
            return False
        if not self._strict_deadline_mode:
            return True

        pos, op, oid = stop
        if op != 1:
            return True

        order = obs["orders"].get(oid)
        if order is None:
            return False

        d_pick = self._dist(shipper.position, pos)
        d_drop = self._get_order_delivery_dist(order)
        if d_pick >= INF or d_drop >= INF:
            return False
        eta = obs["t"] + d_pick + d_drop + 2
        return eta - order.et <= self.max_delivery_delay

    def _update_surge_hotspot_detector(self, obs: dict) -> None:
        orders: Dict[int, Order] = obs["orders"]
        current_t = obs.get("t", 0)
        current_order_ids = set(orders.keys())
        new_order_ids = list(current_order_ids - self._seen_order_ids)
        self._seen_order_ids.update(current_order_ids)
        self.detector.update(current_t, new_order_ids, orders)
        if self.detector.is_surge:
            self._stats["surge_steps"] += 1

    def _hotspot_slots(self, n_idle: int) -> List[Position]:
        hotspots = list(self.detector.predicted_hotspots)
        if n_idle <= 0 or not hotspots:
            return []

        if self.env.N < 100:
            scores = [self.detector.hotspot_scores.get(hp, 1.0) for hp in hotspots]
            total_score = sum(scores) if sum(scores) > 0.0 else 1.0
            slots: List[Position] = []
            for hp, score in zip(hotspots, scores):
                alloc = max(1, int(round(n_idle * (score / total_score))))
                slots.extend([hp] * alloc)
        else:
            max_per_hotspot = max(1, n_idle // len(hotspots))
            slots = []
            for hp in hotspots:
                slots.extend([hp] * max_per_hotspot)

        while len(slots) < n_idle:
            best_hp = max(
                hotspots,
                key=lambda hp: self.detector.hotspot_scores.get(hp, 0.0),
            )
            slots.append(best_hp)

        while len(slots) > n_idle:
            hp_counts = {hp: slots.count(hp) for hp in hotspots}
            removable = [hp for hp, count in hp_counts.items() if count > 1]
            if removable:
                worst_hp = min(
                    removable,
                    key=lambda hp: self.detector.hotspot_scores.get(hp, 0.0),
                )
                slots.remove(worst_hp)
            else:
                slots.pop()

        return slots

    def _assign_hotspot_targets(self, obs: dict) -> None:
        self._shipper_hotspot_assignments = {}
        if not self.detector.is_surge or not self.detector.predicted_hotspots:
            return

        idle_shippers = [
            s for s in obs["shippers"]
            if len(s.bag) == 0 and not self._targets.get(s.id)
        ]
        if not idle_shippers:
            return

        hotspot_slots = self._hotspot_slots(len(idle_shippers))
        if not hotspot_slots:
            return

        cost_matrix = [
            [self._dist(s.position, hp) for hp in hotspot_slots]
            for s in idle_shippers
        ]
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for row, col in zip(row_ind, col_ind):
            row_i, col_i = int(row), int(col)
            if row_i < 0 or col_i < 0:
                continue
            if row_i >= len(idle_shippers) or col_i >= len(hotspot_slots):
                continue
            if cost_matrix[row_i][col_i] >= INF:
                continue
            shipper = idle_shippers[row_i]
            self._shipper_hotspot_assignments[shipper.id] = hotspot_slots[col_i]

        self._stats["hotspot_assignments"] += len(self._shipper_hotspot_assignments)

    def _hotspot_goal_for(self, shipper: Shipper) -> Optional[Position]:
        if not self.detector.is_surge or not self.detector.predicted_hotspots:
            return None

        hotspot = self._shipper_hotspot_assignments.get(shipper.id)
        if hotspot is None:
            hotspot = min(
                self.detector.predicted_hotspots,
                key=lambda hp: self._dist(shipper.position, hp),
            )
        if hotspot == shipper.position:
            return None
        if self._dist(shipper.position, hotspot) >= INF:
            return None
        return hotspot

    def _get_shipper_goal(self, shipper: Shipper, obs: dict) -> Tuple[Position, int]:
        orders: Dict[int, Order] = obs["orders"]
        queue = self._targets.get(shipper.id)

        if queue:
            while queue and not self._is_stop_actionable(queue[0], shipper, obs):
                removed_stop = queue.popleft()
                if removed_stop[1] == 1:
                    self._drop_stale_order_targets(shipper.id, removed_stop[2])
                    queue = self._targets[shipper.id]

            if queue:
                stop = queue[0]
                goal, op, _ = stop
                return goal, op

        current_t = obs["t"]
        carried = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if carried:
            delivery_order = self._select_delivery_order(shipper, orders)
            if delivery_order is None:
                return shipper.position, 0
            self._targets[shipper.id].append(
                ((delivery_order.ex, delivery_order.ey), 2, delivery_order.id)
            )
            stop = self._targets[shipper.id][0]
            return stop[0], stop[1]

        # Shipper is empty
        available_orders = {oid: o for oid, o in orders.items() if oid not in self._reserved}
        pickup_order = self._plan_multi_pickup_route(shipper, available_orders, current_t)
        if pickup_order is None:
            pickup_order = self._select_pickup_v1(shipper, orders, self._reserved, current_t)

        if pickup_order is not None:
            self._reserved.add(pickup_order.id)
            self._targets[shipper.id].append(((pickup_order.sx, pickup_order.sy), 1, pickup_order.id))
            self._targets[shipper.id].append(((pickup_order.ex, pickup_order.ey), 2, pickup_order.id))
            stop = self._targets[shipper.id][0]
            return stop[0], stop[1]

        hotspot_goal = self._hotspot_goal_for(shipper)
        if hotspot_goal is not None:
            return hotspot_goal, 0

        return shipper.position, 0

    def _space_time_astar(
        self,
        start: Position,
        goal: Position,
        constraints: Set[Any],
        max_t: int
    ) -> Optional[List[Position]]:
        """
        Space-Time A* searching for a collision-free path up to max_t.
        Returns a list of positions of length max_t + 1.
        """
        if not constraints:
            moves = self._path(start, goal)
            path = [start]
            curr = start
            for mv in moves[:max_t]:
                curr = valid_next_pos(curr, mv, self.grid)
                path.append(curr)
            while len(path) < max_t + 1:
                path.append(path[-1])
            return path

        # A* Node: (f, g, r, c, t)
        open_list = []
        heapq.heappush(open_list, (self._dist(start, goal), 0, start[0], start[1], 0))
        
        # visited: set of (r, c, t)
        visited = {(start[0], start[1], 0)}
        
        # parent pointers
        parent = {}
        
        best_node = None
        
        while open_list:
            f, g, r, c, t = heapq.heappop(open_list)
            
            if t == max_t:
                best_node = (r, c, t)
                break
                
            for move_str, (dr, dc) in MOVES_CBS.items():
                nr, nc = r + dr, c + dc
                
                # Bounds check if not 'S'
                if move_str != 'S':
                    # Instead of valid_next_pos, just use basic check since grid is fixed
                    if not (0 <= nr < self.env.N and 0 <= nc < self.env.N):
                        continue
                    if self.env.grid[nr][nc] == 1: # Obstacle
                        continue
                
                # Check constraints
                # Vertex constraint: (nr, nc, t+1)
                nt = t + 1
                if (nr, nc, nt) in constraints:
                    continue
                # Edge constraint: (r, c, nr, nc, t) meaning moving from (r,c) to (nr,nc) at time t
                if (r, c, nr, nc, t) in constraints:
                    continue
                    
                state = (nr, nc, nt)
                if state not in visited:
                    visited.add(state)
                    parent[state] = (r, c, t)
                    # heuristic
                    h = self._dist((nr, nc), goal)
                    heapq.heappush(open_list, (nt + h, nt, nr, nc, nt))
                    
        if best_node is None:
            return None
            
        # Reconstruct path
        path = []
        curr = best_node
        while curr:
            path.append((curr[0], curr[1]))
            curr = parent.get(curr)
        path.reverse()
        return path

    def _get_first_conflict(self, paths: Dict[int, List[Position]], max_t: int) -> Optional[Tuple]:
        """
        Find the first conflict among a set of paths.
        Returns:
        - Vertex conflict: ('V', sid1, sid2, r, c, t)
        - Edge conflict: ('E', sid1, sid2, r1, c1, r2, c2, t)
        """
        sids = list(paths.keys())
        for t in range(1, max_t + 1):
            pos_to_sid = {}
            edge_to_sid = {}
            for sid in sids:
                if t < len(paths[sid]):
                    pos = paths[sid][t]
                    # Check vertex conflicts
                    if pos in pos_to_sid:
                        return ('V', pos_to_sid[pos], sid, pos[0], pos[1], t)
                    pos_to_sid[pos] = sid
                    
                    # Check edge conflicts
                    u = paths[sid][t-1]
                    v = pos
                    if u != v:
                        if (v, u) in edge_to_sid:
                            return ('E', edge_to_sid[(v, u)], sid, v[0], v[1], u[0], u[1], t - 1)
                        edge_to_sid[(u, v)] = sid
        return None

    def _cbs(self, start_positions: Dict[int, Position], goals: Dict[int, Position]) -> Optional[Dict[int, List[Position]]]:
        """
        Run Windowed Conflict-Based Search.
        """
        start_time = time.time()
        
        # Root node:
        # constraints: dict {sid: set of constraints}
        # paths: dict {sid: path}
        # cost: int
        
        root_constraints = {sid: set() for sid in start_positions}
        root_paths = {}
        
        for sid in start_positions:
            path = self._space_time_astar(start_positions[sid], goals[sid], set(), self.window_size)
            if path is None:
                return None # No solution even without constraints (e.g., trapped)
            root_paths[sid] = path
            
        root_cost = sum(len(p) for p in root_paths.values())
        
        # Priority queue for CT: (cost, node_id, constraints, paths)
        # Using a simple list as queue for CBS since max nodes is small
        node_id_counter = 0
        open_list = []
        heapq.heappush(open_list, (root_cost, node_id_counter, root_constraints, root_paths))
        node_id_counter += 1
        
        nodes_expanded = 0
        
        while open_list:
            if time.time() - start_time > self.cbs_time_limit or nodes_expanded >= self.cbs_max_nodes:
                return None # Timeout / Max nodes reached
                
            cost, _, constraints, paths = heapq.heappop(open_list)
            nodes_expanded += 1
            
            conflict = self._get_first_conflict(paths, self.window_size)
            if conflict is None:
                return paths # Found a collision-free set of paths
                
            # Branching
            c_type = conflict[0]
            if c_type == 'V':
                _, sid1, sid2, r, c, t = conflict
                
                # Branch 1: sid1 cannot be at (r, c) at time t
                constraints1 = constraints.copy()
                constraints1[sid1] = constraints[sid1].copy()
                constraints1[sid1].add((r, c, t))
                path1 = self._space_time_astar(start_positions[sid1], goals[sid1], constraints1[sid1], self.window_size)
                if path1:
                    paths1 = paths.copy()
                    paths1[sid1] = path1
                    cost1 = sum(len(p) for p in paths1.values())
                    heapq.heappush(open_list, (cost1, node_id_counter, constraints1, paths1))
                    node_id_counter += 1
                    
                # Branch 2: sid2 cannot be at (r, c) at time t
                constraints2 = constraints.copy()
                constraints2[sid2] = constraints[sid2].copy()
                constraints2[sid2].add((r, c, t))
                path2 = self._space_time_astar(start_positions[sid2], goals[sid2], constraints2[sid2], self.window_size)
                if path2:
                    paths2 = paths.copy()
                    paths2[sid2] = path2
                    cost2 = sum(len(p) for p in paths2.values())
                    heapq.heappush(open_list, (cost2, node_id_counter, constraints2, paths2))
                    node_id_counter += 1
                    
            elif c_type == 'E':
                _, sid1, sid2, r1, c1, r2, c2, t = conflict
                
                # Branch 1: sid1 cannot move from (r1, c1) to (r2, c2) at time t
                constraints1 = constraints.copy()
                constraints1[sid1] = constraints[sid1].copy()
                constraints1[sid1].add((r1, c1, r2, c2, t))
                path1 = self._space_time_astar(start_positions[sid1], goals[sid1], constraints1[sid1], self.window_size)
                if path1:
                    paths1 = paths.copy()
                    paths1[sid1] = path1
                    cost1 = sum(len(p) for p in paths1.values())
                    heapq.heappush(open_list, (cost1, node_id_counter, constraints1, paths1))
                    node_id_counter += 1
                    
                # Branch 2: sid2 cannot move from (r2, c2) to (r1, c1) at time t
                constraints2 = constraints.copy()
                constraints2[sid2] = constraints[sid2].copy()
                constraints2[sid2].add((r2, c2, r1, c1, t))
                path2 = self._space_time_astar(start_positions[sid2], goals[sid2], constraints2[sid2], self.window_size)
                if path2:
                    paths2 = paths.copy()
                    paths2[sid2] = path2
                    cost2 = sum(len(p) for p in paths2.values())
                    heapq.heappush(open_list, (cost2, node_id_counter, constraints2, paths2))
                    node_id_counter += 1

    def _should_replan(self, obs: dict) -> bool:
        t = obs["t"]
        if t - self._last_plan_t >= self._replan_interval:
            return True
        if obs.get("new_order_ids"):
            if self.env.G >= 1000 or self.env.C >= 20:
                return len(obs["new_order_ids"]) >= 5 and t - self._last_plan_t >= 10
            if self.env.N >= 50:
                return t - self._last_plan_t >= 5
            return True

        has_unassigned = any(not o.picked and not o.delivered for o in obs["orders"].values())
        if has_unassigned:
            if self.env.G >= 1000 or self.env.C >= 20:
                return False
            if self.env.N >= 50 and t - self._last_plan_t < 10:
                return False
            for s in obs["shippers"]:
                if not self._targets.get(s.id):
                    return True

        return False

    def _active_orders(self, obs: dict) -> List[Order]:
        if self._strict_deadline_mode:
            orders: Dict[int, Order] = obs["orders"]
            shippers: List[Shipper] = obs["shippers"]
            obs_t = obs["t"]
            candidates: List[Order] = []
            metrics: Dict[int, Tuple[int, int, float]] = {}
            for order in orders.values():
                if order.picked or order.delivered:
                    continue
                best_eta = INF
                best_score = 0.0
                for shipper in shippers:
                    if not shipper.can_carry(order, orders):
                        continue
                    d_pick = self._dist(shipper.position, (order.sx, order.sy))
                    d_drop = self._get_order_delivery_dist(order)
                    if d_pick >= INF or d_drop >= INF:
                        continue
                    eta = obs_t + d_pick + d_drop
                    best_eta = min(best_eta, eta)
                    reward = delivery_reward(order, eta, self.env.T)
                    best_score = max(best_score, reward / max(d_pick + d_drop, 1))
                if best_eta >= INF or best_score <= 0:
                    continue
                feasible_rank = 0 if best_eta - order.et <= self.max_delivery_delay else 1
                metrics[order.id] = (feasible_rank, best_eta, best_score)
                candidates.append(order)

            candidates.sort(
                key=lambda o: (
                    metrics[o.id][0],
                    -metrics[o.id][2],
                    o.et,
                    metrics[o.id][1],
                    -o.p,
                    o.id,
                )
            )
        else:
            candidates = super()._active_orders(obs)
        max_active = self._max_active_override or max(50, self.env.C * 3)
        return candidates[:max_active]

    def _eta_pickup(
        self,
        pos: Position,
        order: Order,
        elapsed: int,
        obs_t: int,
        d_pick: int,
    ) -> float:
        if not self._strict_deadline_mode:
            return super()._eta_pickup(pos, order, elapsed, obs_t, d_pick)

        d_drop = self._get_order_delivery_dist(order)
        if d_pick >= INF or d_drop >= INF:
            return 0.0

        eta_delivery = obs_t + elapsed + d_pick + d_drop
        reward = delivery_reward(order, eta_delivery, self.env.T)
        if reward <= 0:
            return 0.0

        lateness = eta_delivery - order.et
        if lateness > self.max_delivery_delay:
            return 0.0

        slack = max(order.et - eta_delivery, 0)
        urgency = 1.0 / max(order.et - obs_t, 1)
        return reward / (d_pick + d_drop + 1) + 12.0 * urgency + 0.05 * order.p + 0.002 * slack

    def _eta_delivery(
        self,
        order: Order,
        elapsed: int,
        obs_t: int,
        d: int,
    ) -> float:
        if not self._strict_deadline_mode:
            return super()._eta_delivery(order, elapsed, obs_t, d)

        if d >= INF:
            return 0.0
        arrival = obs_t + elapsed + d
        reward = delivery_reward(order, arrival, self.env.T)
        if arrival <= order.et:
            urgency = 2.0 + 1.0 / max(order.et - obs_t, 1)
        else:
            urgency = max(0.2, 1.0 - (arrival - order.et) / max(self.env.T, 1))
        return max(0.001, reward * urgency / (d + 1))

    def _replan(self, obs: dict) -> None:
        self._stats["aco_calls"] += 1
        return super()._replan(obs)

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
                    # Filter active orders to only the closest ones
                    if self.env.G >= 1000 or self.env.C >= 20:
                        k_closest = min(len(active_orders), self._pickup_candidate_limit())
                    else:
                        k_closest = max(40, len(active_orders) // 2)
                    if len(active_orders) > k_closest:
                        shipper_active_orders = heapq.nsmallest(
                            k_closest,
                            active_orders,
                            key=lambda o: abs(pos[0] - o.sx) + abs(pos[1] - o.sy)
                        )
                    else:
                        shipper_active_orders = active_orders

                    for order in shipper_active_orders:
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

    def _action_from_goal(self, shipper: Shipper, goal: Position, op: int) -> Action:
        return self._navigate_to(shipper.position, goal, op)

    def _action_from_cbs_path(
        self,
        sid: int,
        curr_pos: Position,
        path: List[Position],
        goal: Position,
        op: int,
    ) -> Action:
        if len(path) > 1:
            next_pos = path[1]
            dr = next_pos[0] - curr_pos[0]
            dc = next_pos[1] - curr_pos[1]
            move = MOVE_TO_STR.get((dr, dc), "S")
        else:
            move = "S"

        next_pos = valid_next_pos(curr_pos, move, self.grid)
        if move != "S" and next_pos == goal:
            return (move, op)
        if move == "S" and curr_pos == goal:
            return ("S", op)
        return (move, 0)

    def _independent_actions(
        self,
        shippers: List[Shipper],
        goals: Dict[int, Position],
        ops: Dict[int, int],
    ) -> Dict[int, Action]:
        actions: Dict[int, Action] = {}
        for shipper in sorted(shippers, key=lambda s: s.id):
            actions[shipper.id] = self._action_from_goal(
                shipper,
                goals[shipper.id],
                ops[shipper.id],
            )
        return actions

    def _predicted_conflict_pairs(
        self,
        shippers: List[Shipper],
        actions: Dict[int, Action],
    ) -> Set[Tuple[int, int]]:
        positions = {s.id: s.position for s in shippers}
        desired = {
            s.id: valid_next_pos(s.position, actions.get(s.id, ("S", 0))[0], self.grid)
            for s in shippers
        }

        pairs: Set[Tuple[int, int]] = set()
        by_target: Dict[Position, List[int]] = {}
        for sid, pos in desired.items():
            by_target.setdefault(pos, []).append(sid)
        for sids in by_target.values():
            if len(sids) > 1:
                for i in range(len(sids)):
                    for j in range(i + 1, len(sids)):
                        pairs.add(tuple(sorted((sids[i], sids[j]))))

        for s1 in shippers:
            for s2 in shippers:
                if s1.id >= s2.id:
                    continue
                if desired[s1.id] == positions[s2.id] and desired[s2.id] == positions[s1.id]:
                    pairs.add((s1.id, s2.id))
                elif desired[s1.id] == positions[s2.id] and desired[s2.id] == positions[s2.id]:
                    pairs.add((s1.id, s2.id))
                elif desired[s2.id] == positions[s1.id] and desired[s1.id] == positions[s1.id]:
                    pairs.add((s1.id, s2.id))
        return pairs

    def _conflict_groups(self, shippers: List[Shipper], actions: Dict[int, Action]) -> List[Set[int]]:
        pairs = self._predicted_conflict_pairs(shippers, actions)
        if not pairs:
            return []

        parent = {s.id: s.id for s in shippers}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for a, b in pairs:
            union(a, b)

        groups: Dict[int, Set[int]] = {}
        for a, b in pairs:
            groups.setdefault(find(a), set()).update((a, b))
        return list(groups.values())

    def _is_serious_conflict_group(
        self,
        group: Set[int],
        shipper_by_id: Dict[int, Shipper],
        ops: Dict[int, int],
    ) -> bool:
        for sid in group:
            shipper = shipper_by_id[sid]
            if shipper.bag or ops.get(sid) in {1, 2}:
                return True
        return False

    def _run_cbs_counted(
        self,
        start_positions: Dict[int, Position],
        goals: Dict[int, Position],
        group_call: bool = False,
    ) -> Optional[Dict[int, List[Position]]]:
        self._stats["cbs_calls"] += 1
        if group_call:
            self._stats["group_cbs_calls"] += 1
        paths = self._cbs(start_positions, goals)
        if paths is None:
            self._stats["cbs_failed"] += 1
        else:
            self._stats["cbs_success"] += 1
        return paths

    def _adaptive_actions(
        self,
        obs: dict,
        goals: Dict[int, Position],
        ops: Dict[int, int],
        start_positions: Dict[int, Position],
    ) -> Dict[int, Action]:
        shippers: List[Shipper] = obs["shippers"]
        shipper_by_id = {s.id: s for s in shippers}

        if self._cbs_mode == "full":
            cbs_paths = self._run_cbs_counted(start_positions, goals)
            if cbs_paths is not None:
                return {
                    sid: self._action_from_cbs_path(
                        sid,
                        start_positions[sid],
                        cbs_paths[sid],
                        goals[sid],
                        ops[sid],
                    )
                    for sid in start_positions
                }

        actions = self._independent_actions(shippers, goals, ops)
        if self._cbs_mode != "group":
            return actions

        self._stats["local_steps"] += 1
        for group in self._conflict_groups(shippers, actions):
            if len(group) > self._group_cbs_max_size:
                continue
            if (self.env.G >= 1000 or self.env.C >= 20) and not self._is_serious_conflict_group(group, shipper_by_id, ops):
                continue

            sub_starts = {sid: start_positions[sid] for sid in group}
            sub_goals = {sid: goals[sid] for sid in group}
            cbs_paths = self._run_cbs_counted(sub_starts, sub_goals, group_call=True)
            if cbs_paths is None:
                continue
            for sid in group:
                actions[sid] = self._action_from_cbs_path(
                    sid,
                    start_positions[sid],
                    cbs_paths[sid],
                    goals[sid],
                    ops[sid],
                )
        return actions

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        self._targets = {s.id: deque() for s in obs["shippers"]}
        self._reserved = set()
        self._last_plan_t = -self._replan_interval
        self._seen_order_ids = set()
        self.detector = OnlineSurgeHotspotDetector(
            self.env.N, len(obs["shippers"]), obs["G"], self.env.T, self.grid
        )
        self._shipper_hotspot_assignments = {}

        while not obs.get("done", False):
            self._update_surge_hotspot_detector(obs)
            if self._should_replan(obs):
                self._replan(obs)

            self._assign_hotspot_targets(obs)
            self._reserved = set()
            goals: Dict[int, Position] = {}
            ops: Dict[int, int] = {}
            start_positions: Dict[int, Position] = {}
            
            # Prioritize empty shippers first to claim pickup tasks
            shippers_sorted = sorted(obs["shippers"], key=lambda s: (len(s.bag) > 0, s.id))
            for shipper in shippers_sorted:
                goal, op = self._get_shipper_goal(shipper, obs)
                goals[shipper.id] = goal
                ops[shipper.id] = op
                start_positions[shipper.id] = shipper.position
                
            actions = self._adaptive_actions(obs, goals, ops, start_positions)

            actions = self._resolve_deadlocks(obs["shippers"], actions)
            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        result = self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
        result["mapd_cbs_stats"] = dict(self._stats)
        result["cbs_mode"] = self._cbs_mode
        result["replan_interval"] = self._replan_interval
        return result
