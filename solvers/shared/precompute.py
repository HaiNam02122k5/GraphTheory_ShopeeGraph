from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple

_PRECOMPUTE_CACHE: Dict[Tuple[Tuple[int, ...], ...], GridPrecompute] = {}

def get_precompute(grid: List[List[int]]) -> GridPrecompute:
    grid_hash = tuple(tuple(row) for row in grid)
    if grid_hash not in _PRECOMPUTE_CACHE:
        _PRECOMPUTE_CACHE[grid_hash] = GridPrecompute(grid)
    return _PRECOMPUTE_CACHE[grid_hash]

class GridPrecompute:
    def __init__(self, grid: List[List[int]]):
        self.grid = grid
        self.N = len(grid)
        self.components: Dict[Tuple[int, int], int] = {}
        self.bottleneck_set: Set[Tuple[int, int]] = set()
        
        self._compute_components()
        self._compute_bottlenecks()

    def _compute_components(self):
        visited = set()
        comp_id = 0
        N = self.N
        grid = self.grid
        for r in range(N):
            for c in range(N):
                if grid[r][c] == 0 and (r, c) not in visited:
                    # Run BFS to label this component
                    queue = deque([(r, c)])
                    visited.add((r, c))
                    while queue:
                        curr = queue.popleft()
                        self.components[curr] = comp_id
                        cr, cc = curr
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] == 0:
                                if (nr, nc) not in visited:
                                    visited.add((nr, nc))
                                    queue.append((nr, nc))
                    comp_id += 1

    def _compute_bottlenecks(self):
        N = self.N
        grid = self.grid
        for r in range(N):
            for c in range(N):
                if grid[r][c] == 0:
                    free_neighbors = 0
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] == 0:
                            free_neighbors += 1
                    if free_neighbors <= 2:
                        self.bottleneck_set.add((r, c))

    def are_connected(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        cid_a = self.components.get(a)
        cid_b = self.components.get(b)
        if cid_a is None or cid_b is None:
            return False
        return cid_a == cid_b

    def is_bottleneck(self, pos: Tuple[int, int]) -> bool:
        return pos in self.bottleneck_set
