from __future__ import annotations
import time
import heapq
from collections import deque
from typing import Dict, List, Tuple, Set, Optional, Any

from env import DeliveryEnv, Order, Shipper, valid_next_pos
from solvers.solver import default_result
from solvers.aco_solver import ACOSolver, INF

Position = Tuple[int, int]
Move = str
Action = Tuple[Move, int]

MOVES_CBS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1), "S": (0, 0)}
MOVE_TO_STR = {(-1, 0): "U", (1, 0): "D", (0, -1): "L", (0, 1): "R", (0, 0): "S"}

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

    def _get_shipper_goal(self, shipper: Shipper, obs: dict) -> Tuple[Position, int]:
        orders: Dict[int, Order] = obs["orders"]
        queue = self._targets.get(shipper.id)

        if queue:
            while queue and not self._is_stop_actionable(queue[0], shipper, obs):
                queue.popleft()

            if queue:
                stop = queue[0]
                goal, op, _ = stop

                if op == 2:
                    return goal, op

                return goal, op

        carried = [
            orders[oid]
            for oid in shipper.bag
            if oid in orders and not orders[oid].delivered
        ]
        if carried:
            best = min(
                carried,
                key=lambda o: (self._dist(shipper.position, (o.ex, o.ey)), o.et, -o.p, o.id),
            )
            return (best.ex, best.ey), 2

        # Greedy pick fallback
        candidates = []
        obs_t = obs["t"]
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
            if obs_t + d_pick + d_drop >= order.et + self.env.T:
                continue
            candidates.append(order)

        if not candidates:
            return shipper.position, 0

        best = min(
            candidates,
            key=lambda o: (
                self._dist(shipper.position, (o.sx, o.sy)),
                -o.p,
                o.et,
                o.id,
            ),
        )
        self._reserved.add(best.id)
        return (best.sx, best.sy), 1

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
        # A* Node: (f, g, r, c, t)
        open_list = []
        heapq.heappush(open_list, (self._dist(start, goal), 0, start[0], start[1], 0))
        
        # visited: (r, c, t) -> g
        visited = {(start[0], start[1], 0): 0}
        
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
                if (nr, nc, t + 1) in constraints:
                    continue
                # Edge constraint: (r, c, nr, nc, t) meaning moving from (r,c) to (nr,nc) at time t
                if (r, c, nr, nc, t) in constraints:
                    continue
                    
                nt = t + 1
                if (nr, nc, nt) not in visited or visited[(nr, nc, nt)] > g + 1:
                    visited[(nr, nc, nt)] = g + 1
                    parent[(nr, nc, nt)] = (r, c, t)
                    # heuristic
                    h = self._dist((nr, nc), goal)
                    heapq.heappush(open_list, (g + 1 + h, g + 1, nr, nc, nt))
                    
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
            # Check vertex conflicts
            pos_to_sid = {}
            for sid in sids:
                if t < len(paths[sid]):
                    pos = paths[sid][t]
                    if pos in pos_to_sid:
                        return ('V', pos_to_sid[pos], sid, pos[0], pos[1], t)
                    pos_to_sid[pos] = sid
            
            # Check edge conflicts
            for i in range(len(sids)):
                for j in range(i + 1, len(sids)):
                    sid1, sid2 = sids[i], sids[j]
                    if t < len(paths[sid1]) and t < len(paths[sid2]):
                        u1 = paths[sid1][t-1]
                        v1 = paths[sid1][t]
                        u2 = paths[sid2][t-1]
                        v2 = paths[sid2][t]
                        if u1 == v2 and v1 == u2 and u1 != v1:
                            return ('E', sid1, sid2, u1[0], u1[1], v1[0], v1[1], t - 1)
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
                constraints1 = {k: set(v) for k, v in constraints.items()}
                constraints1[sid1].add((r, c, t))
                path1 = self._space_time_astar(start_positions[sid1], goals[sid1], constraints1[sid1], self.window_size)
                if path1:
                    paths1 = {k: list(v) for k, v in paths.items()}
                    paths1[sid1] = path1
                    cost1 = sum(len(p) for p in paths1.values())
                    heapq.heappush(open_list, (cost1, node_id_counter, constraints1, paths1))
                    node_id_counter += 1
                    
                # Branch 2: sid2 cannot be at (r, c) at time t
                constraints2 = {k: set(v) for k, v in constraints.items()}
                constraints2[sid2].add((r, c, t))
                path2 = self._space_time_astar(start_positions[sid2], goals[sid2], constraints2[sid2], self.window_size)
                if path2:
                    paths2 = {k: list(v) for k, v in paths.items()}
                    paths2[sid2] = path2
                    cost2 = sum(len(p) for p in paths2.values())
                    heapq.heappush(open_list, (cost2, node_id_counter, constraints2, paths2))
                    node_id_counter += 1
                    
            elif c_type == 'E':
                _, sid1, sid2, r1, c1, r2, c2, t = conflict
                
                # Branch 1: sid1 cannot move from (r1, c1) to (r2, c2) at time t
                constraints1 = {k: set(v) for k, v in constraints.items()}
                constraints1[sid1].add((r1, c1, r2, c2, t))
                path1 = self._space_time_astar(start_positions[sid1], goals[sid1], constraints1[sid1], self.window_size)
                if path1:
                    paths1 = {k: list(v) for k, v in paths.items()}
                    paths1[sid1] = path1
                    cost1 = sum(len(p) for p in paths1.values())
                    heapq.heappush(open_list, (cost1, node_id_counter, constraints1, paths1))
                    node_id_counter += 1
                    
                # Branch 2: sid2 cannot move from (r2, c2) to (r1, c1) at time t
                constraints2 = {k: set(v) for k, v in constraints.items()}
                constraints2[sid2].add((r2, c2, r1, c1, t))
                path2 = self._space_time_astar(start_positions[sid2], goals[sid2], constraints2[sid2], self.window_size)
                if path2:
                    paths2 = {k: list(v) for k, v in paths.items()}
                    paths2[sid2] = path2
                    cost2 = sum(len(p) for p in paths2.values())
                    heapq.heappush(open_list, (cost2, node_id_counter, constraints2, paths2))
                    node_id_counter += 1

        return None # No solution found within limits

    def run(self) -> dict:
        start_time = time.time()
        obs = self.env.reset()

        self._targets = {s.id: deque() for s in obs["shippers"]}
        self._reserved = set()
        self._last_plan_t = -self._replan_interval

        # Store the planned paths for execution
        planned_paths: Dict[int, List[Position]] = {}

        while not obs.get("done", False):
            if self._should_replan(obs):
                self._replan(obs)

            self._reserved = set()
            goals: Dict[int, Position] = {}
            ops: Dict[int, int] = {}
            start_positions: Dict[int, Position] = {}
            
            for shipper in sorted(obs["shippers"], key=lambda s: s.id):
                goal, op = self._get_shipper_goal(shipper, obs)
                goals[shipper.id] = goal
                ops[shipper.id] = op
                start_positions[shipper.id] = shipper.position
                
            # Use Windowed CBS to find collision-free next steps
            cbs_paths = self._cbs(start_positions, goals)
            
            actions: Dict[int, Action] = {}
            
            if cbs_paths is not None:
                # Execution from CBS plan
                for sid in start_positions:
                    if len(cbs_paths[sid]) > 1:
                        next_pos = cbs_paths[sid][1]
                        curr_pos = start_positions[sid]
                        dr, dc = next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1]
                        move_str = MOVE_TO_STR.get((dr, dc), "S")
                    else:
                        move_str = "S"
                        
                    # Check if goal reached to apply op
                    if move_str != "S" and valid_next_pos(curr_pos, move_str, self.grid) == goals[sid]:
                        actions[sid] = (move_str, ops[sid])
                    elif move_str == "S" and curr_pos == goals[sid]:
                        actions[sid] = ("S", ops[sid])
                    else:
                        actions[sid] = (move_str, 0)
            else:
                # Fallback: execute standard greedy paths independently
                for shipper in sorted(obs["shippers"], key=lambda s: s.id):
                    # We can use the ACO _navigate_to for fallback
                    goal = goals[shipper.id]
                    op = ops[shipper.id]
                    actions[shipper.id] = self._navigate_to(shipper.position, goal, op)

            obs, _, done, _ = self.env.step(actions)
            if done:
                break

        return self.env.result(
            self.method_name,
            elapsed_sec=time.time() - start_time,
        )
