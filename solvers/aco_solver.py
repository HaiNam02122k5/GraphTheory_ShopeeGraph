from __future__ import annotations

import heapq
import os
import random
import time
from collections import defaultdict, deque
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

from env import (
    DeliveryEnv,
    Order,
    Shipper,
    delivery_reward,
    is_valid_cell,
    move_cost,
    valid_next_pos,
)
from solvers.solver import Solver

Move = str
Position = Tuple[int, int]
Action = Tuple[Move, object]

DIR_MOVES: Tuple[Move, ...] = ("U", "D", "L", "R")
REVERSE_MOVE = {"U": "D", "D": "U", "L": "R", "R": "L"}
INF = 10**9


class ACOSolver(Solver):
    method_name = "ACO"

    def __init__(self, env: DeliveryEnv):
        super().__init__(env)

        self.rng = random.Random(42)
        tuned_defaults = self._phase1_tuned_defaults()
        self.zone_size = self._env_int("ACO_ZONE_SIZE", int(tuned_defaults.get("zone_size", max(2, self.env.N // 8))), 1, max(1, self.env.N))
        self.num_ants = self._env_int("ACO_NUM_ANTS", int(tuned_defaults.get("num_ants", 10 if self.env.C <= 8 else 6)), 1, 128)
        self.path_ants = 48 if self.env.N <= 30 else 14
        self.max_candidates = self._env_int("ACO_MAX_CANDIDATES", int(tuned_defaults.get("max_candidates", 35 if self.env.G <= 300 else 25)), 1, max(1, self.env.G))
        self.assign_mode = self._env_choice("ACO_ASSIGN_MODE", str(tuned_defaults.get("assign_mode", "auto")), {"auto", "ant", "greedy", "beam"})
        self.move_mode = self._env_choice("ACO_MOVE_MODE", str(tuned_defaults.get("move_mode", "astar_occupied_deadlock")), {"astar_current", "astar_deadlock", "bfs_cached", "astar_occupied", "astar_occupied_deadlock", "bfs_deadlock"}, "astar_current")
        self.active_assign_mode = "ant" if self.assign_mode == "auto" else self.assign_mode
        self.assign_beam_top_k = 4
        self.assign_beam_width = 12
        self.reserve_busy_pickups = self._env_bool("ACO_RESERVE_BUSY_PICKUPS", bool(tuned_defaults.get("reserve_busy_pickups", False)))
        self.reposition_mode = self._env_choice("ACO_REPOSITION_MODE", str(tuned_defaults.get("reposition_mode", "pressure_current")), {"none", "pressure_current"})
        self.pressure_zone_mode = self._env_choice("ACO_PRESSURE_ZONE_MODE", str(tuned_defaults.get("pressure_zone_mode", "current")), {"small", "current", "large"})
        self.pressure_zone_size = self._pressure_zone_size(self.pressure_zone_mode)
        self.pressure_decay = self._env_float("ACO_PRESSURE_DECAY", float(tuned_defaults.get("pressure_decay", 0.97)), 0.50, 0.999)

        self.alpha = self._env_float("ACO_ALPHA", float(tuned_defaults.get("alpha", 0.70)), 0.0, 5.0)
        self.beta = self._env_float("ACO_BETA", float(tuned_defaults.get("beta", 1.25)), 0.1, 8.0)
        self.rho = self._env_float("ACO_RHO", float(tuned_defaults.get("rho", 0.030)), 0.0, 0.5)

        self.q = 0.080
        self.min_tau = 0.45
        self.max_tau = 3.00

        self.pressure: DefaultDict[Position, float] = defaultdict(float)
        self.tau_pickup: DefaultDict[Tuple[Position, int], float] = defaultdict(lambda: 1.0)
        self.tau_flow: DefaultDict[Tuple[Position, Position, int], float] = defaultdict(lambda: 1.0)
        self.tau_move: DefaultDict[Tuple[Position, Move], float] = defaultdict(lambda: 1.0)
        self.congestion: DefaultDict[Position, float] = defaultdict(float)

        self._known_orders: Dict[int, Order] = {}
        self._delivered_updates: Set[int] = set()
        self._last_positions: Dict[int, deque[Position]] = defaultdict(lambda: deque(maxlen=8))
        self._stuck_count: DefaultDict[int, int] = defaultdict(int)
        self._last_distance_to_target: Dict[int, int] = {}
        self._last_goal: Dict[int, Position] = {}
        self._last_reposition_zone: Dict[int, Position] = {}
        self._metrics: DefaultDict[str, float] = defaultdict(float)
        self._astar_cache: Dict[Tuple[Position, Position], Move] = {}
        self._adj: Dict[Position, List[Tuple[Move, Position]]] = {}
        self._bfs_cache: Dict[Position, Tuple[Dict[Position, int], Dict[Position, Move]]] = {}
        self._bfs_cache_order: deque[Position] = deque()
        self._dist_cache: Dict[Position, Dict[Position, int]] = {}
        self._dist_cache_order: deque[Position] = deque()
        self._cache_limit = 700 if self.env.N <= 30 else 260 if self.env.N <= 60 else 140
        self._pair_dist_cache: Dict[Tuple[Position, Position], int] = {}
        self._order_delivery_dist: Dict[int, int] = {}
        self._build_adjacency_list()

    def _phase1_tuned_defaults(self) -> Dict[str, object]:
        return {
            "C1": {
                "assign_mode": "ant",
                "move_mode": "astar_deadlock",
                "reposition_mode": "pressure_current",
                "zone_size": 8,
                "pressure_zone_mode": "large",
                "num_ants": 3,
                "max_candidates": 10,
                "rho": 0.15,
                "reserve_busy_pickups": True,
            },
            "C2": {
                "assign_mode": "greedy",
                "move_mode": "bfs_deadlock",
                "reposition_mode": "pressure_current",
                "zone_size": 6,
                "pressure_decay": 0.80,
                "num_ants": 32,
                "max_candidates": 4,
                "alpha": 0.70,
                "beta": 1.25,
            },
            "C3": {
                "assign_mode": "greedy",
                "move_mode": "bfs_deadlock",
                "reposition_mode": "pressure_current",
                "pressure_decay": 0.88,
                "reserve_busy_pickups": True,
            },
            "C4": {
                "assign_mode": "ant",
                "move_mode": "astar_occupied_deadlock",
                "reposition_mode": "pressure_current",
                "zone_size": 5,
                "pressure_decay": 0.99,
                "num_ants": 8,
                "alpha": 0.50,
                "beta": 1.50,
            },
            "C5": {
                "assign_mode": "beam",
                "move_mode": "astar_occupied_deadlock",
                "reposition_mode": "pressure_current",
                "zone_size": 8,
                "pressure_zone_mode": "current",
                "pressure_decay": 0.90,
                "max_candidates": 45,
                "alpha": 0.50,
                "beta": 1.75,
                "rho": 0.0,
            },
            "C6": {
                "assign_mode": "ant",
                "move_mode": "astar_occupied_deadlock",
                "reposition_mode": "pressure_current",
                "zone_size": 2,
                "num_ants": 24,
                "beta": 2.50,
            },
        }.get(str(getattr(self.env, "config_name", "")), {})

    @staticmethod
    def _env_int(key: str, default: int, min_value: int, max_value: int) -> int:
        raw = os.environ.get(key)
        if raw is None:
            return default
        try:
            value = int(raw)
        except ValueError:
            return default
        return min(max_value, max(min_value, value))

    @staticmethod
    def _env_bool(key: str, default: bool) -> bool:
        raw = os.environ.get(key)
        if raw is None:
            return default
        return raw.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_float(key: str, default: float, min_value: float, max_value: float) -> float:
        raw = os.environ.get(key)
        if raw is None:
            return default
        try:
            value = float(raw)
        except ValueError:
            return default
        return min(max_value, max(min_value, value))

    @staticmethod
    def _env_choice(key: str, default: str, choices: Set[str], fallback: Optional[str] = None) -> str:
        value = os.environ.get(key, default).strip().lower()
        return value if value in choices else fallback or default

    def _pressure_zone_size(self, mode: str) -> int:
        if mode == "small":
            return max(1, self.zone_size // 2)
        if mode == "large":
            return max(self.zone_size + 1, self.env.N // 5)
        return self.zone_size

    def _zone(self, pos: Position) -> Position:
        return pos[0] // self.zone_size, pos[1] // self.zone_size

    def _pressure_zone(self, pos: Position) -> Position:
        return pos[0] // self.pressure_zone_size, pos[1] // self.pressure_zone_size

    @staticmethod
    def _manhattan(a: Position, b: Position) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _build_adjacency_list(self) -> None:
        for r in range(self.env.N):
            for c in range(self.env.N):
                pos = (r, c)
                if not is_valid_cell(pos, self.grid):
                    continue
                neighbors: List[Tuple[Move, Position]] = []
                for move in DIR_MOVES:
                    nxt = valid_next_pos(pos, move, self.grid)
                    if nxt != pos:
                        neighbors.append((move, nxt))
                self._adj[pos] = neighbors

    def _valid_moves(self, pos: Position) -> List[Tuple[Move, Position]]:
        out: List[Tuple[Move, Position]] = [("S", pos)]
        out.extend(self._adj.get(pos, []))
        return out

    def _remember_cache(self, cache: dict, order: deque, key: Position, value: object) -> None:
        if key not in cache:
            order.append(key)
        cache[key] = value
        while len(order) > self._cache_limit:
            old = order.popleft()
            if old != key:
                cache.pop(old, None)

    def _save_dist_cache(self, start: Position, dist_map: Dict[Position, int]) -> None:
        self._remember_cache(self._dist_cache, self._dist_cache_order, start, dist_map)

    def _save_bfs_cache(self, start: Position, dist_map: Dict[Position, int], next_move_map: Dict[Position, Move]) -> None:
        self._remember_cache(
            self._bfs_cache,
            self._bfs_cache_order,
            start,
            (dist_map, next_move_map),
        )
        self._save_dist_cache(start, dist_map)

    def _bfs_from(self, start: Position) -> Tuple[Dict[Position, int], Dict[Position, Move]]:
        cached = self._bfs_cache.get(start)
        if cached is not None:
            return cached

        dist_map: Dict[Position, int] = {start: 0}
        next_move_map: Dict[Position, Move] = {start: "S"}
        if not is_valid_cell(start, self.grid):
            self._save_bfs_cache(start, dist_map, next_move_map)
            return dist_map, next_move_map

        q: deque[Position] = deque()
        for move, nxt in self._adj.get(start, []):
            dist_map[nxt] = 1
            next_move_map[nxt] = move
            q.append(nxt)

        while q:
            pos = q.popleft()
            next_d = dist_map[pos] + 1
            first_move = next_move_map[pos]
            for _, nxt in self._adj.get(pos, []):
                if nxt in dist_map:
                    continue
                dist_map[nxt] = next_d
                next_move_map[nxt] = first_move
                q.append(nxt)

        self._save_bfs_cache(start, dist_map, next_move_map)
        return dist_map, next_move_map

    def _distances_from(self, start: Position) -> Dict[Position, int]:
        cached = self._dist_cache.get(start)
        if cached is not None:
            return cached
        return self._bfs_from(start)[0]

    def _astar_search(self, start: Position, goal: Position, blocked: Optional[Set[Position]] = None) -> Tuple[int, Move]:
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return INF, "S"

        open_heap: List[Tuple[int, int, Position]] = [(self._manhattan(start, goal), 0, start)]
        best_g: Dict[Position, int] = {start: 0}
        parent: Dict[Position, Tuple[Position, Move]] = {}

        while open_heap:
            _, g_score, pos = heapq.heappop(open_heap)
            if g_score != best_g.get(pos):
                continue
            if pos == goal:
                first_move = "S"
                while pos != start:
                    pos, first_move = parent[pos]
                return g_score, first_move

            for move, nxt in self._adj.get(pos, []):
                if blocked and nxt != goal and nxt in blocked:
                    continue
                next_g = g_score + 1
                if next_g >= best_g.get(nxt, INF):
                    continue
                best_g[nxt] = next_g
                parent[nxt] = (pos, move)
                heapq.heappush(
                    open_heap,
                    (next_g + self._manhattan(nxt, goal), next_g, nxt),
                )

        return INF, "S"

    def _astar_distance(self, start: Position, goal: Position) -> int:
        cached = self._pair_dist_cache.get((start, goal))
        if cached is not None:
            return cached
        dist, _ = self._astar_search(start, goal)
        self._pair_dist_cache[(start, goal)] = dist
        self._pair_dist_cache[(goal, start)] = dist
        return dist

    def _distance(self, start: Position, goal: Position) -> int:
        if start == goal:
            return 0
        cached = self._dist_cache.get(start)
        if cached is not None:
            return cached.get(goal, INF)
        cached = self._dist_cache.get(goal)
        if cached is not None:
            return cached.get(start, INF)
        cached_pair = self._pair_dist_cache.get((start, goal))
        if cached_pair is not None:
            return cached_pair
        if self.env.N <= 60:
            dist = self._distances_from(start).get(goal, INF)
            self._pair_dist_cache[(start, goal)] = dist
            self._pair_dist_cache[(goal, start)] = dist
            return dist
        return self._astar_distance(start, goal)

    def _get_order_delivery_dist(self, order: Order) -> int:
        cached = self._order_delivery_dist.get(order.id)
        if cached is not None:
            return cached
        dist = self._distance((order.sx, order.sy), (order.ex, order.ey))
        self._order_delivery_dist[order.id] = dist
        return dist

    def _flow_key(self, order: Order) -> Tuple[Position, Position, int]:
        return self._zone((order.sx, order.sy)), self._zone((order.ex, order.ey)), order.p

    def _pickup_tau(self, order: Order) -> float:
        return self.tau_pickup[(self._zone((order.sx, order.sy)), order.p)]

    def _flow_tau(self, order: Order) -> float:
        return self.tau_flow[self._flow_key(order)]

    def _current_weight(self, shipper: Shipper, orders: Dict[int, Order]) -> float:
        return sum(orders[oid].w for oid in shipper.bag if oid in orders)

    def _can_carry_virtual(self, shipper: Shipper, order: Order, orders: Dict[int, Order], extra_order_ids: Optional[Set[int]] = None) -> bool:
        if order.picked or order.delivered:
            return False
        extra_order_ids = extra_order_ids or set()
        slots = len(shipper.bag) + len(extra_order_ids)
        weight = self._current_weight(shipper, orders)
        for oid in extra_order_ids:
            if oid in orders:
                weight += orders[oid].w
        return slots < shipper.K_max and weight + order.w <= shipper.W_max

    def _update_pressure(self, obs: dict) -> None:
        for key in list(self.pressure.keys()):
            self.pressure[key] *= self.pressure_decay
            if self.pressure[key] < 0.02:
                del self.pressure[key]

        orders: Dict[int, Order] = obs["orders"]
        current_t = obs.get("t", 0)
        new_order_ids = obs.get("new_order_ids", [])
        for oid in new_order_ids:
            order = orders.get(oid)
            if order is None:
                continue
            slack = max(order.et - current_t, 1)
            deadline_weight = 1.0 + min(3.0, 30.0 / slack)
            self.pressure[self._pressure_zone((order.sx, order.sy))] += (
                order.p * deadline_weight
            )

    def _evaporate(self) -> None:
        for table in (self.tau_pickup, self.tau_flow, self.tau_move):
            for key in list(table.keys()):
                val = 1.0 + (table[key] - 1.0) * (1.0 - self.rho)
                if abs(val - 1.0) < 0.002:
                    del table[key]
                else:
                    table[key] = min(self.max_tau, max(self.min_tau, val))

        for key in list(self.congestion.keys()):
            self.congestion[key] *= 0.90
            if self.congestion[key] < 0.03:
                del self.congestion[key]

    def _observe_deliveries(self, before: Dict[int, Order], after: Dict[int, Order], delivery_t: int) -> None:
        active_after = set(after)
        for oid, order in before.items():
            if oid in active_after or oid in self._delivered_updates:
                continue
            if not order.picked or order.carrier < 0:
                continue

            reward = delivery_reward(order, delivery_t, self.env.T)
            if reward <= 0.0:
                continue

            self._delivered_updates.add(oid)
            pickup_zone = self._zone((order.sx, order.sy))
            flow_key = self._flow_key(order)
            delta = self.q * (reward / 30.0) * (1.0 + 0.12 * (order.p - 1))
            self.tau_pickup[(pickup_zone, order.p)] = min(
                self.max_tau,
                self.tau_pickup[(pickup_zone, order.p)] + delta,
            )
            self.tau_flow[flow_key] = min(self.max_tau, self.tau_flow[flow_key] + delta)

    def _order_utility(self, shipper: Shipper, order: Order, orders: Dict[int, Order], current_t: int, from_pos: Optional[Position] = None, d_pick: Optional[int] = None, d_deliver: Optional[int] = None) -> float:
        pos = from_pos or shipper.position
        pickup = (order.sx, order.sy)
        dropoff = (order.ex, order.ey)
        if d_pick is None:
            d_pick = self._distance(pos, pickup)
        if d_deliver is None:
            d_deliver = self._get_order_delivery_dist(order)
        if d_pick >= INF or d_deliver >= INF:
            return 0.001
        total_dist = max(1, d_pick + d_deliver)
        eta_t = current_t + d_pick + d_deliver
        reward = delivery_reward(order, eta_t, self.env.T)

        if eta_t <= order.et:
            slack = max(order.et - eta_t, 0)
            deadline_factor = 1.0 + min(2.5, 20.0 / max(slack + 1, 1))
            # Deadline urgency helps when detours are cheap; very large maps need throughput.
            if self.env.N <= 15 or 20 <= self.env.N < 30:
                urgency_bonus = 10.0 / max(order.et - current_t, 1)
            else:
                urgency_bonus = 0.0
        else:
            deadline_factor = max(0.10, 1.0 - (eta_t - order.et) / max(self.env.T, 1))
            urgency_bonus = 0.0

        carried_w = self._current_weight(shipper, orders)
        move_cost = 0.01 * total_dist * (
            1.0 + (carried_w + order.w) / max(shipper.W_max, 1.0)
        )
        zone = self._pressure_zone(pickup)
        pressure = 1.0 + min(2.0, self.pressure.get(zone, 0.0) / 8.0)
        pheromone = (self._pickup_tau(order) * self._flow_tau(order)) ** 0.5
        load_fit = 1.0 + 0.15 * (shipper.K_max - len(shipper.bag))

        reward_density = (reward - move_cost) / total_dist + urgency_bonus
        return max(0.001, reward_density * deadline_factor * pressure * load_fit * pheromone)

    def _candidate_orders(self, shipper: Shipper, orders: Dict[int, Order], reserved: Set[int], current_t: int) -> List[Order]:
        scored: List[Tuple[float, int, Order]] = []
        pickup_dist = self._distances_from(shipper.position)
        reachable: List[Tuple[float, int, Order]] = []
        for order in orders.values():
            if order.id in reserved or order.picked or order.delivered or not self._can_carry_virtual(shipper, order, orders):
                continue
            d_pick = pickup_dist.get((order.sx, order.sy), INF)
            if d_pick >= INF:
                continue
            rough_eta = current_t + d_pick
            rough_slack = max(order.et - rough_eta, -self.env.T)
            rough_score = (
                order.p * 10.0
                + self._pickup_tau(order)
                + self._flow_tau(order)
                + max(0.0, rough_slack) / max(order.et, 1)
                - 0.04 * d_pick
            )
            reachable.append((rough_score, -d_pick, order))

        if self.env.N > 60:
            reachable.sort(key=lambda item: (item[0], item[1], item[2].p), reverse=True)
            eval_orders = reachable[: max(self.max_candidates * 3, 60)]
        else:
            eval_orders = reachable

        for _, neg_d_pick, order in eval_orders:
            d_pick = -neg_d_pick
            d_deliver = self._get_order_delivery_dist(order)
            if d_deliver >= INF:
                continue
            utility = self._order_utility(shipper, order, orders, current_t, d_pick=d_pick, d_deliver=d_deliver)
            scored.append((utility, -d_pick, order))

        scored.sort(key=lambda item: (item[0], item[1], item[2].p), reverse=True)
        return [order for _, _, order in scored[: self.max_candidates]]

    def _weighted_pick(self, items: List[Tuple[Order, float]]) -> Optional[Order]:
        if not items:
            return None
        total = sum(max(weight, 0.0) for _, weight in items)
        if total <= 0.0:
            return max(items, key=lambda item: item[1])[0]

        r = self.rng.random() * total
        acc = 0.0
        for order, weight in items:
            acc += max(weight, 0.0)
            if acc >= r:
                return order
        return items[-1][0]

    def _assignment_candidates(self, shippers: Iterable[Shipper], orders: Dict[int, Order], current_t: int, reserved_initial: Optional[Set[int]] = None) -> Dict[int, List[Tuple[float, int, Order]]]:
        reserved = set(reserved_initial or set())
        candidates_by_shipper: Dict[int, List[Tuple[float, int, Order]]] = {}
        for shipper in shippers:
            candidates: List[Tuple[float, int, Order]] = []
            for order in self._candidate_orders(shipper, orders, reserved, current_t):
                pickup = (order.sx, order.sy)
                d_pick = self._distance(shipper.position, pickup)
                if d_pick >= INF:
                    continue
                d_deliver = self._get_order_delivery_dist(order)
                if d_deliver >= INF:
                    continue
                utility = self._order_utility(shipper, order, orders, current_t, d_pick=d_pick, d_deliver=d_deliver)
                candidates.append((utility, d_pick, order))

            candidates.sort(key=lambda item: (item[0], -item[1], item[2].p, -item[2].et, -item[2].id), reverse=True)
            candidates_by_shipper[shipper.id] = candidates
        return candidates_by_shipper

    def _greedy_assign_pickups(self, shippers: Iterable[Shipper], orders: Dict[int, Order], current_t: int, reserved_initial: Optional[Set[int]] = None) -> Dict[int, Order]:
        shippers = list(shippers)
        if not shippers:
            return {}

        candidates_by_shipper = self._assignment_candidates(shippers, orders, current_t, reserved_initial)
        edges: List[Tuple[float, int, int, int, int, Shipper, Order]] = [
            (utility, -d_pick, order.p, -order.et, -shipper.id, shipper, order)
            for shipper in shippers
            for utility, d_pick, order in candidates_by_shipper.get(shipper.id, [])
        ]

        edges.sort(key=lambda item: item[:5], reverse=True)
        used_orders = set(reserved_initial or set())
        used_shippers: Set[int] = set()
        assignment: Dict[int, Order] = {}
        for _, _, _, _, _, shipper, order in edges:
            if shipper.id in used_shippers or order.id in used_orders:
                continue
            assignment[shipper.id] = order
            used_shippers.add(shipper.id)
            used_orders.add(order.id)
        return assignment

    def _beam_assign_pickups(self, shippers: Iterable[Shipper], orders: Dict[int, Order], current_t: int, reserved_initial: Optional[Set[int]] = None) -> Dict[int, Order]:
        shippers = list(shippers)
        if not shippers:
            return {}

        candidates_by_shipper = self._assignment_candidates(shippers, orders, current_t, reserved_initial)

        def shipper_order_key(shipper: Shipper) -> Tuple[int, float, int]:
            candidates = candidates_by_shipper.get(shipper.id, [])
            best_utility = candidates[0][0] if candidates else 0.0
            return len(candidates), -best_utility, shipper.id

        ordered_shippers = sorted(shippers, key=shipper_order_key)

        beams: List[Tuple[float, Dict[int, Order], Set[int]]] = [(0.0, {}, set(reserved_initial or set()))]
        for shipper in ordered_shippers:
            options = candidates_by_shipper.get(shipper.id, [])[: self.assign_beam_top_k]
            next_beams: List[Tuple[float, Dict[int, Order], Set[int]]] = []
            for score, assignment, used_orders in beams:
                next_beams.append((score, assignment, used_orders))
                for utility, _, order in options:
                    if order.id in used_orders:
                        continue
                    next_beams.append((score + utility, {**assignment, shipper.id: order}, used_orders | {order.id}))

            next_beams.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
            beams = next_beams[: self.assign_beam_width]

        return max(beams, key=lambda item: (item[0], len(item[1])))[1]

    def _assign_pickups(self, shippers: Iterable[Shipper], orders: Dict[int, Order], current_t: int, reserved_initial: Optional[Set[int]] = None) -> Dict[int, Order]:
        assigner = {"greedy": self._greedy_assign_pickups, "beam": self._beam_assign_pickups}.get(
            self.active_assign_mode,
            self._ant_assign_pickups,
        )
        return assigner(shippers, orders, current_t, reserved_initial)

    def _ant_assign_pickups(self, shippers: Iterable[Shipper], orders: Dict[int, Order], current_t: int, reserved_initial: Optional[Set[int]] = None) -> Dict[int, Order]:
        shippers = list(shippers)
        if not shippers:
            return {}

        best_score = -INF
        best_assignment: Dict[int, Order] = {}

        for ant_idx in range(self.num_ants):
            reserved: Set[int] = set(reserved_initial or set())
            assignment: Dict[int, Order] = {}
            total_score = 0.0
            ant_shippers = list(shippers)
            if ant_idx > 0:
                self.rng.shuffle(ant_shippers)

            for shipper in ant_shippers:
                weighted = [
                    (order, ((self._pickup_tau(order) * self._flow_tau(order)) ** self.alpha) * (self._order_utility(shipper, order, orders, current_t) ** self.beta))
                    for order in self._candidate_orders(shipper, orders, reserved, current_t)
                ]
                chosen = max(weighted, key=lambda item: item[1])[0] if ant_idx == 0 and weighted else self._weighted_pick(weighted)

                if chosen is None:
                    continue
                reserved.add(chosen.id)
                assignment[shipper.id] = chosen
                total_score += self._order_utility(shipper, chosen, orders, current_t)

            if total_score > best_score:
                best_score, best_assignment = total_score, assignment

        return best_assignment

    def _best_delivery_order(self, shipper: Shipper, orders: Dict[int, Order], current_t: int) -> Optional[Order]:
        carried = [orders[oid] for oid in shipper.bag if oid in orders]
        if not carried:
            return None

        delivery_dist = self._distances_from(shipper.position)

        def key(order: Order) -> Tuple[float, int, int]:
            d = delivery_dist.get((order.ex, order.ey), INF)
            if d >= INF:
                return (-INF, -order.et, order.p)
            eta_t = current_t + d
            reward_density = delivery_reward(order, eta_t, self.env.T) / max(1, d)
            on_time_bonus = 100.0 if eta_t <= order.et else 0.0
            return (reward_density + on_time_bonus, -order.et, order.p)

        best = max(carried, key=key)
        if delivery_dist.get((best.ex, best.ey), INF) >= INF:
            return None
        return best

    def _estimate_route_eval(self, shipper: Shipper, pickup_orders: List[Order], bag_orders: List[Order], current_t: int) -> Optional[Tuple[Dict[int, int], float]]:
        t = current_t
        curr = shipper.position
        carried = list(bag_orders)
        carried_weight = sum(order.w for order in carried)
        total_reward = 0.0
        total_move_cost = 0.0
        delivery_times: Dict[int, int] = {}

        for order in pickup_orders:
            dist = self._distance(curr, (order.sx, order.sy))
            if dist >= INF:
                return None
            total_move_cost += dist * move_cost(carried_weight, shipper.W_max)
            t += dist
            curr = (order.sx, order.sy)
            carried.append(order)
            carried_weight += order.w

        remaining = list(carried)
        while remaining:
            def delivery_key(order: Order) -> Tuple[int, int, int, int, int]:
                dist = self._distance(curr, (order.ex, order.ey))
                if dist >= INF:
                    return (2, INF, order.et, -order.p, order.id)
                eta_t = t + dist
                if eta_t <= order.et:
                    return (0, order.et, dist, -order.p, order.id)
                return (1, dist, order.et, -order.p, order.id)

            order = min(remaining, key=delivery_key)
            dist = self._distance(curr, (order.ex, order.ey))
            if dist >= INF:
                return None
            total_move_cost += dist * move_cost(carried_weight, shipper.W_max)
            t += dist
            curr = (order.ex, order.ey)
            total_reward += delivery_reward(order, t, self.env.T)
            delivery_times[order.id] = t
            carried_weight = max(0.0, carried_weight - order.w)
            remaining.remove(order)

        return delivery_times, total_reward + total_move_cost

    def _opportunistic_pickup(self, shipper: Shipper, delivery_order: Order, orders: Dict[int, Order], current_t: int, reserved: Set[int]) -> Optional[Order]:
        if len(shipper.bag) >= shipper.K_max:
            return None

        delivery_target = (delivery_order.ex, delivery_order.ey)
        current_dist = self._distances_from(shipper.position)
        direct_delivery_dist = current_dist.get(delivery_target, INF)
        if direct_delivery_dist >= INF:
            return None
        direct_delivery_t = current_t + direct_delivery_dist

        bag_orders = [orders[oid] for oid in shipper.bag if oid in orders and not orders[oid].delivered]
        if (baseline := self._estimate_route_eval(shipper, [], bag_orders, current_t)) is None:
            return None
        baseline_times, baseline_net = baseline

        use_original_gate = self.env.N in (15, 20)
        max_pick_dist = 10 if self.env.N >= 20 else 5 if use_original_gate else 30 if self.env.N == 12 else 2
        best_order = None
        best_score = 0.0

        for order in orders.values():
            if (
                order.id in reserved
                or order.picked
                or order.delivered
                or not self._can_carry_virtual(shipper, order, orders)
            ):
                continue

            pickup = (order.sx, order.sy)
            d_pick = current_dist.get(pickup, INF)
            if d_pick > max_pick_dist:
                continue

            d_deliver = self._get_order_delivery_dist(order)
            if d_deliver >= INF:
                continue

            detour = 0
            delayed_delivery_t = direct_delivery_t
            if use_original_gate or self.env.N != 12:
                pickup_to_delivery_target = self._distance(pickup, delivery_target)
                if pickup_to_delivery_target >= INF:
                    continue
                delayed_delivery_t = current_t + d_pick + pickup_to_delivery_target
                detour = max(0, delayed_delivery_t - direct_delivery_t)

            if use_original_gate:
                if self.env.N >= 20:
                    allowed_delay = 4 if delivery_order.p == 3 else 14
                else:
                    allowed_delay = 8
                if delayed_delivery_t > delivery_order.et + allowed_delay:
                    continue

                candidate_delay = 18 if self.env.N >= 20 else 10
                if current_t + d_pick + d_deliver > order.et + candidate_delay:
                    continue
            else:
                if current_t + d_pick + d_deliver > order.et:
                    continue
                if self.env.N != 12 and detour > 2:
                    continue

            if (route_eval := self._estimate_route_eval(shipper, [order], bag_orders, current_t)) is None:
                continue
            new_times, new_net = route_eval
            if any(new_times.get(bag_order.id, INF) > bag_order.et for bag_order in bag_orders):
                continue
            if not use_original_gate and (
                any(
                    new_times.get(bag_order.id, INF) > baseline_times.get(bag_order.id, INF)
                    and baseline_times.get(bag_order.id, INF) <= bag_order.et
                    for bag_order in bag_orders
                )
                or new_times.get(order.id, INF) > order.et
            ):
                continue

            net_gain = new_net - baseline_net
            if net_gain <= 0.05:
                continue

            zone = self._pressure_zone(pickup)
            pressure = 1.0 + min(2.0, self.pressure.get(zone, 0.0) / 8.0)
            pheromone = (self._pickup_tau(order) * self._flow_tau(order)) ** 0.5
            if use_original_gate:
                utility = self._order_utility(shipper, order, orders, current_t, d_pick=d_pick, d_deliver=d_deliver)
                score = (utility - detour * (0.22 if self.env.N >= 20 else 0.35) + net_gain) * pressure * pheromone
            else:
                slack = max(order.et - new_times.get(order.id, order.et), 0)
                urgency = 1.0 + min(1.5, 20.0 / max(slack + 1, 1))
                score = net_gain * pressure * pheromone * urgency
            if score > best_score:
                best_score = score
                best_order = order

        return best_order

    def _best_pressure_target(self, shipper: Shipper) -> Optional[Position]:
        if not self.pressure:
            return None
        best_zone = None
        best_score = -INF
        for zone, pressure in self.pressure.items():
            center = (
                min(
                    self.env.N - 1,
                    zone[0] * self.pressure_zone_size + self.pressure_zone_size // 2,
                ),
                min(
                    self.env.N - 1,
                    zone[1] * self.pressure_zone_size + self.pressure_zone_size // 2,
                ),
            )
            if not is_valid_cell(center, self.grid):
                continue
            dist = max(1, self._manhattan(shipper.position, center))
            score = pressure / dist - self.congestion.get(zone, 0.0)
            if score > best_score:
                best_score = score
                best_zone = center
        return best_zone

    def _move_towards(self, shipper: Shipper, target: Position, occupied_next: Set[Position]) -> Tuple[Move, Position]:
        pos = shipper.position
        if pos == target:
            return "S", pos

        move = self._aco_next_move(shipper, target, occupied_next)
        nxt = valid_next_pos(pos, move, self.grid)
        if nxt in occupied_next and nxt != target:
            fallback = self._best_local_escape(shipper, target, occupied_next)
            return fallback
        return move, nxt

    def _rollout_move_weights(self, current: Position, target: Position, visited: Set[Position], occupied_next: Set[Position]) -> List[Tuple[Move, Position, float]]:
        old_d = self._manhattan(current, target)
        weighted: List[Tuple[Move, Position, float]] = []

        for move, nxt in self._valid_moves(current):
            if move == "S" and len(weighted) > 0:
                continue
            new_d = self._manhattan(nxt, target)
            progress = old_d - new_d
            zone = self._zone(nxt)
            revisit = 0.35 if nxt in visited else 0.0
            collision = 4.0 if nxt in occupied_next and nxt != target else 0.0
            congestion = self.congestion.get(zone, 0.0)
            tau = self.tau_move[(self._zone(current), move)]
            closeness = 1.0 / max(1, new_d)
            raw = (
                1.0
                + progress * 1.35
                + closeness * 4.0
                + tau * 0.55
                - revisit
                - collision
                - congestion
            )
            if progress < 0:
                raw *= 0.72
            weighted.append((move, nxt, max(0.001, raw)))

        return weighted

    def _sample_rollout_step(self, weighted: List[Tuple[Move, Position, float]], greedy: bool) -> Tuple[Move, Position]:
        if greedy:
            return max(weighted, key=lambda item: item[2])[:2]

        total = sum(item[2] for item in weighted)
        r = self.rng.random() * total
        acc = 0.0
        for move, nxt, weight in weighted:
            acc += weight
            if acc >= r:
                return move, nxt
        return weighted[-1][:2]

    def _aco_next_move(self, shipper: Shipper, target: Position, occupied_next: Set[Position]) -> Move:
        if self.move_mode in {"bfs_cached", "bfs_deadlock"}:
            return self._bfs_next_move(shipper, target, occupied_next)
        if self.env.N >= 15:
            dynamic = self.move_mode in {"astar_occupied", "astar_occupied_deadlock"}
            return self._astar_next_move(shipper, target, occupied_next, dynamic_occupied=dynamic)

        start = shipper.position
        direct_d = self._manhattan(start, target)
        max_steps = min(max(self.env.N * 4, direct_d * 4 + 20), 260)
        best_path: List[Move] = []
        best_score = -INF

        for ant_idx in range(self.path_ants):
            current, visited, path, reached = start, {start}, [], False

            for _ in range(max_steps):
                weighted = self._rollout_move_weights(current, target, visited, occupied_next if not path else set())
                if not weighted:
                    break
                move, nxt = self._sample_rollout_step(weighted, greedy=(ant_idx == 0))
                if move == "S" and current != target:
                    break
                path.append(move)
                current = nxt
                visited.add(current)
                if current == target:
                    reached = True
                    break

            score = -self._manhattan(current, target) * 3.0 - len(path) * 0.05
            score += 1000.0 - len(path) if reached else 0.0
            if path and path[0] != "S" and valid_next_pos(start, path[0], self.grid) in self._last_positions[shipper.id]:
                score -= 1.0

            if score > best_score and path:
                best_score, best_path = score, path

        return best_path[0] if best_path else self._best_local_escape(shipper, target, occupied_next)[0]

    def _bfs_next_move(self, shipper: Shipper, target: Position, occupied_next: Set[Position]) -> Move:
        start = shipper.position
        if start == target:
            return "S"

        _, next_move_map = self._bfs_from(start)
        move = next_move_map.get(target, "S")
        if move == "S":
            move, _ = self._best_local_escape(shipper, target, occupied_next)
            return move

        nxt = valid_next_pos(start, move, self.grid)
        if nxt in occupied_next and nxt != target:
            move, _ = self._best_local_escape(shipper, target, occupied_next)
        return move

    def _astar_next_move(self, shipper: Shipper, target: Position, occupied_next: Set[Position], dynamic_occupied: bool = False) -> Move:
        start = shipper.position
        if start == target:
            return "S"

        cached = None if dynamic_occupied else self._astar_cache.get((start, target))
        if cached is not None:
            nxt = valid_next_pos(start, cached, self.grid)
            if nxt not in occupied_next or nxt == target:
                return cached

        blocked = occupied_next if dynamic_occupied else None
        dist, first_move = self._astar_search(start, target, blocked)
        if dist >= INF:
            move, _ = self._best_local_escape(shipper, target, occupied_next)
            return move

        if not dynamic_occupied:
            self._astar_cache[(start, target)] = first_move
        first_pos = valid_next_pos(start, first_move, self.grid)
        if first_pos in occupied_next and first_pos != target:
            move, _ = self._best_local_escape(shipper, target, occupied_next)
            return move
        return first_move

    def _best_local_escape(self, shipper: Shipper, target: Position, occupied_next: Set[Position]) -> Tuple[Move, Position]:
        pos = shipper.position
        old_d = self._manhattan(pos, target)
        last_positions = self._last_positions[shipper.id]
        scored: List[Tuple[float, Move, Position]] = []

        for move, nxt in self._valid_moves(pos):
            new_d = self._manhattan(nxt, target)
            collision = 10.0 if nxt in occupied_next and nxt != target else 0.0
            revisit = 0.8 if nxt in last_positions else 0.0
            wait = 0.5 if move == "S" else 0.0
            score = (
                (old_d - new_d) * 2.0
                - collision
                - revisit
                - wait
                - self.congestion.get(self._zone(nxt), 0.0)
            )
            scored.append((score, move, nxt))

        scored.sort(key=lambda item: item[0], reverse=True)
        return scored[0][1], scored[0][2]

    def _uses_deadlock_resolver(self) -> bool:
        if self.move_mode == "bfs_deadlock":
            return True
        if self.move_mode in {"astar_deadlock", "astar_occupied_deadlock"}:
            return self.env.N >= 15
        return False

    def _resolve_deadlocks(self, shippers: List[Shipper], actions: Dict[int, Action]) -> Dict[int, Action]:
        resolved = dict(actions)
        positions = {shipper.id: shipper.position for shipper in shippers}
        occupied_now = set(positions.values())
        desired = {
            shipper.id: valid_next_pos(shipper.position, resolved.get(shipper.id, ("S", 0))[0], self.grid)
            for shipper in shippers
        }

        for first in shippers:
            for second in shippers:
                if first.id >= second.id:
                    continue

                first_pos = positions[first.id]
                second_pos = positions[second.id]
                first_des = desired[first.id]
                second_des = desired[second.id]

                if first_des != second_pos or second_des != first_pos or first_pos == second_pos:
                    continue

                evader = second.id
                evader_pos = second_pos
                other_pos = first_pos
                reserved_next = {pos for sid, pos in desired.items() if sid != evader}

                for move in DIR_MOVES:
                    nxt = valid_next_pos(evader_pos, move, self.grid)
                    if (
                        nxt != evader_pos
                        and nxt != other_pos
                        and nxt not in occupied_now
                        and nxt not in reserved_next
                    ):
                        resolved[evader] = (move, 0)
                        desired[evader] = nxt
                        break
                else:
                    reverse_move = REVERSE_MOVE.get(actions.get(evader, ("S", 0))[0])
                    if reverse_move is not None:
                        nxt = valid_next_pos(evader_pos, reverse_move, self.grid)
                        if nxt != evader_pos and nxt not in occupied_now and nxt not in reserved_next:
                            resolved[evader] = (reverse_move, 0)
                            desired[evader] = nxt
                            continue
                    resolved[evader] = ("S", 0)
                    desired[evader] = evader_pos

        return resolved

    def _pickup_action(self, shipper: Shipper, order: Order, occupied_next: Set[Position]) -> Tuple[Action, Position, Position]:
        target = (order.sx, order.sy)
        last_reposition_zone = self._last_reposition_zone.pop(shipper.id, None)
        if last_reposition_zone is not None:
            self._metrics["post_reposition_pickup_assignments"] += 1
            if last_reposition_zone == self._pressure_zone(target):
                self._metrics["post_reposition_pickup_hits"] += 1
        move, nxt = self._move_towards(shipper, target, occupied_next)
        op = 1 if nxt == target else 0
        return (move, op), nxt, target

    def _delivery_action(self, shipper: Shipper, order: Order, occupied_next: Set[Position]) -> Tuple[Action, Position, Position]:
        target = (order.ex, order.ey)
        self._last_reposition_zone.pop(shipper.id, None)
        move, nxt = self._move_towards(shipper, target, occupied_next)
        op = 2 if nxt == target else 0
        return (move, op), nxt, target

    def _reposition_action(self, shipper: Shipper, occupied_next: Set[Position]) -> Tuple[Action, Position, Position]:
        if self.reposition_mode == "none":
            self._last_reposition_zone.pop(shipper.id, None)
            return ("S", 0), shipper.position, shipper.position
        target = self._best_pressure_target(shipper)
        if target is None or target == shipper.position:
            return ("S", 0), shipper.position, shipper.position
        move, nxt = self._move_towards(shipper, target, occupied_next)
        self._metrics["reposition_attempts"] += 1
        if move != "S":
            self._metrics["reposition_steps"] += 1
            self._last_reposition_zone[shipper.id] = self._pressure_zone(target)
        return (move, 0), nxt, target

    def _update_motion_learning(self, shipper: Shipper, action: Action, target: Position, next_pos: Position) -> None:
        move = action[0]
        old_d = self._last_distance_to_target.get(
            shipper.id,
            self._manhattan(shipper.position, target),
        )
        new_d = self._manhattan(next_pos, target)
        if target != self._last_goal.get(shipper.id):
            self._stuck_count[shipper.id] = 0
        elif new_d >= old_d and move != "S":
            self._stuck_count[shipper.id] += 1
        else:
            self._stuck_count[shipper.id] = max(0, self._stuck_count[shipper.id] - 1)

        z = self._zone(shipper.position)
        if new_d < old_d:
            self.tau_move[(z, move)] = min(self.max_tau, self.tau_move[(z, move)] + 0.015)
        elif move != "S":
            self.tau_move[(z, move)] = max(self.min_tau, self.tau_move[(z, move)] - 0.020)

        if self._stuck_count[shipper.id] >= 4:
            self.congestion[self._zone(shipper.position)] += 0.25

        self._last_distance_to_target[shipper.id] = new_d
        self._last_goal[shipper.id] = target
        self._last_positions[shipper.id].append(next_pos)

    def _commit_action(self, shipper: Shipper, action_state: Tuple[Action, Position, Position], actions: Dict[int, Action], occupied_next: Set[Position], learn: bool = True) -> None:
        action, nxt, target = action_state
        actions[shipper.id] = action
        occupied_next.add(nxt)
        if learn:
            self._update_motion_learning(shipper, action, target, nxt)

    def _decide_actions(self, obs: dict) -> Dict[int, Action]:
        orders: Dict[int, Order] = obs["orders"]
        shippers: List[Shipper] = obs["shippers"]
        current_t = obs.get("t", 0)

        self._known_orders.update(orders)
        self._update_pressure(obs)
        self._evaporate()

        actions: Dict[int, Action] = {}
        occupied_next: Set[Position] = set()
        reserved_pickups: Set[int] = set()

        for shipper in shippers:
            if any((order := orders.get(oid)) and shipper.position == (order.ex, order.ey) for oid in shipper.bag):
                self._commit_action(shipper, (("S", 2), shipper.position, shipper.position), actions, occupied_next)

        busy = [s for s in shippers if s.id not in actions and s.bag]
        empty = [s for s in shippers if s.id not in actions and not s.bag]

        for shipper in sorted(busy, key=lambda s: (len(s.bag), s.id)):
            delivery_order = self._best_delivery_order(shipper, orders, current_t)
            if delivery_order is None:
                self._commit_action(
                    shipper,
                    (("S", 0), shipper.position, shipper.position),
                    actions,
                    occupied_next,
                    learn=False,
                )
                continue
            opportunistic = self._opportunistic_pickup(shipper, delivery_order, orders, current_t, reserved_pickups)
            if opportunistic is not None:
                reserved_pickups.add(opportunistic.id)
                action_state = self._pickup_action(shipper, opportunistic, occupied_next)
            else:
                action_state = self._delivery_action(shipper, delivery_order, occupied_next)
            self._commit_action(shipper, action_state, actions, occupied_next)

        assignments = self._assign_pickups(
            empty,
            orders,
            current_t,
            reserved_initial=reserved_pickups if self.reserve_busy_pickups else set(),
        )
        for shipper in sorted(empty, key=lambda s: s.id):
            assigned = assignments.get(shipper.id)
            action_state = (
                self._pickup_action(shipper, assigned, occupied_next)
                if assigned is not None
                else self._reposition_action(shipper, occupied_next)
            )
            self._commit_action(shipper, action_state, actions, occupied_next)

        if self._uses_deadlock_resolver():
            actions = self._resolve_deadlocks(shippers, actions)

        return actions

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()
        seen_order_ids: Set[int] = set(obs["orders"].keys())
        self._known_orders = dict(obs["orders"])

        while not obs.get("done", False):
            seen_order_ids.update(obs["orders"].keys())
            no_more_generation = len(seen_order_ids) == obs["G"]
            all_bags_empty = all(len(s.bag) == 0 for s in obs["shippers"])
            no_available_pickups = all(o.picked for o in obs["orders"].values())
            if no_more_generation and all_bags_empty and no_available_pickups:
                break

            before_orders = dict(obs["orders"])
            delivery_t = obs.get("t", 0)
            actions = self._decide_actions(obs)
            obs, _, done, _ = self.env.step(actions)
            self._observe_deliveries(before_orders, obs["orders"], delivery_t)
            if done:
                break

        result = self.env.result(self.method_name, elapsed_sec=time.time() - start_time)
        result["aco_pressure"] = {
            "reposition_mode": self.reposition_mode,
            "zone_mode": self.pressure_zone_mode,
            "zone_size": self.pressure_zone_size,
            "decay": self.pressure_decay,
        }
        result["aco_params"] = {
            "num_ants": self.num_ants,
            "max_candidates": self.max_candidates,
            "alpha": self.alpha,
            "beta": self.beta,
            "rho": self.rho,
            "zone_size": self.zone_size,
            "pressure_decay": self.pressure_decay,
        }
        for key, value in self._metrics.items():
            result[f"aco_{key}"] = round(value, 4)
        return result
