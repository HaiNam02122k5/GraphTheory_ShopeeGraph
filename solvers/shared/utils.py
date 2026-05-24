from typing import List, Tuple, Any

ALPHA = {1: 1.0, 2: 2.0, 3: 3.0}
BETA  = {1: 0.1, 2: 0.3, 3: 0.5}
GAMMA = 1.0
DIRS = {"S": (0, 0), "U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

def manhattan(r1: int, c1: int, r2: int, c2: int) -> int:
    """Khoảng cách Manhattan giữa hai ô lưới."""
    return abs(r1 - r2) + abs(c1 - c2)

def is_valid_cell(pos: Tuple[int, int], grid: List[List[int]]) -> bool:
    """True nếu pos trong bản đồ và không phải ô vật cản (grid[r][c] == 0)."""
    r, c = pos
    return 0 <= r < len(grid) and 0 <= c < len(grid[0]) and grid[r][c] == 0

def next_pos(pos: Tuple[int, int], move: str) -> Tuple[int, int]:
    """Tọa độ kế tiếp theo hướng move, không kiểm tra hợp lệ."""
    dr, dc = DIRS.get(move, (0, 0))
    return pos[0] + dr, pos[1] + dc

def valid_next_pos(pos: Tuple[int, int], move: str, grid: List[List[int]]) -> Tuple[int, int]:
    """Tọa độ kế tiếp sau move; giữ nguyên pos nếu ô đích bị chặn hoặc ra ngoài."""
    nxt = next_pos(pos, move)
    return nxt if is_valid_cell(nxt, grid) else pos

def r_base(w: float) -> float:
    """Phần thưởng cơ bản theo khối lượng đơn hàng."""
    if w <= 0.2:  return 4.0
    if w <= 3.0:  return 10.0
    if w <= 10.0: return 15.0
    if w <= 30.0: return 20.0
    return 30.0

def delivery_reward(order: Any, t_delivery: int, T: int) -> float:
    """Tính phần thưởng giao hàng theo công thức đề bài (có/không có penalty trễ hạn)."""
    rb = r_base(order.w)
    if t_delivery <= order.et:
        bonus = max(0.0, (order.et - t_delivery) / max(order.et, 1))
        return ALPHA[order.p] * rb * (1.0 + bonus)
    factor = max(0.0, 1.0 - (t_delivery - order.et) / max(T, 1))
    return BETA[order.p] * rb * factor

def move_cost(w_carried: float, w_max: float) -> float:
    """Chi phí di chuyển một bước theo tải trọng hiện tại."""
    return -0.01 * (1.0 + GAMMA * w_carried / max(w_max, 1.0))
