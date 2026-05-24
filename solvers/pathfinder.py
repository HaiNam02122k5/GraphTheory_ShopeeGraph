from __future__ import annotations

import time
from collections import deque
from typing import Dict, List, Optional, Tuple, Set

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
        
        # Fallback CPU BFS caches (always initialized for safety/testing)
        self._bfs_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], Tuple[int, List[str]]] = {}
        self._distance_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {}
        self._next_move_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], str] = {}
        
        self.has_torch = self._precompute_with_torch(grid)

    def _precompute_with_torch(self, grid: List[List[int]]) -> bool:
        try:
            import torch
        except ImportError:
            return False

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

        # 4. Multi-Source Parallel BFS loop
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

        # 5. Extract matrices to CPU numpy for fast lookups
        self.dist_matrix = dist[:V, :V].cpu().numpy()
        self.next_move_matrix = first_move[:V, :V].cpu().numpy()
        self.adj_cpu = adj[:V].cpu().numpy()
        return True

    def _fallback_bfs(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Tuple[int, List[str]]:
        if start == goal:
            return 0, []
        from env import is_valid_cell, valid_next_pos
        if not is_valid_cell(start, self.grid) or not is_valid_cell(goal, self.grid):
            return 10**9, []

        parent: Dict[Tuple[int, int], Tuple[Optional[Tuple[int, int]], str]] = {start: (None, "S")}
        queue = deque([start])
        moves_list = ("U", "D", "L", "R")

        found = False
        while queue:
            cur = queue.popleft()
            if cur == goal:
                found = True
                break
            for mv in moves_list:
                nxt = valid_next_pos(cur, mv, self.grid)
                if nxt != cur and nxt not in parent:
                    parent[nxt] = (cur, mv)
                    queue.append(nxt)

        if not found or goal not in parent:
            return 10**9, []

        moves = []
        cur = goal
        while cur != start:
            prev, mv = parent[cur]
            moves.append(mv)
            cur = prev  # type: ignore
        moves.reverse()
        return len(moves), moves

    def dist(self, start: Tuple[int, int], goal: Tuple[int, int]) -> int:
        if self.has_torch:
            if start == goal:
                return 0
            s_idx = self.cell_to_idx.get(start)
            g_idx = self.cell_to_idx.get(goal)
            if s_idx is None or g_idx is None:
                return 10**9
            d = self.dist_matrix[s_idx, g_idx]
            return 10**9 if d >= 9999 else int(d)
        else:
            key = (start, goal)
            if key not in self._distance_cache:
                d, _ = self._fallback_bfs(start, goal)
                self._distance_cache[key] = d
            return self._distance_cache[key]

    def path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[str]:
        if self.has_torch:
            if start == goal:
                return []
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
            key = (start, goal)
            if key not in self._bfs_cache:
                d, p = self._fallback_bfs(start, goal)
                self._bfs_cache[key] = (d, p)
            return self._bfs_cache[key][1]

    def next_move(self, start: Tuple[int, int], goal: Tuple[int, int]) -> str:
        if self.has_torch:
            if start == goal:
                return "S"
            s_idx = self.cell_to_idx.get(start)
            g_idx = self.cell_to_idx.get(goal)
            if s_idx is None or g_idx is None:
                return "S"
            move_idx = self.next_move_matrix[s_idx, g_idx]
            if 0 <= move_idx < 4:
                return ["U", "D", "L", "R"][move_idx]
            return "S"
        else:
            key = (start, goal)
            if key not in self._next_move_cache:
                p = self.path(start, goal)
                self._next_move_cache[key] = p[0] if p else "S"
            return self._next_move_cache[key]
