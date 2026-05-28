from typing import List, Tuple, Dict
from env import Order
from solvers.shared.utils import manhattan

class OnlineSurgeHotspotDetector:
    """
    Ước lượng Surge và Hotspot trực tuyến từ Observation (tuân thủ rule không đọc Env/Config trực tiếp).
    """
    def __init__(self, N: int, C: int, G: int, T: int, grid: List[List[int]]):
        self.N = N
        self.C = C
        self.G = G
        self.T = T
        self.grid = grid
        
        self.lambda0 = G / max(T, 1)
        self.history_counts = []
        self.pickup_history: List[Tuple[int, Tuple[int, int]]] = []  # list of (t, pos)
        self.pickup_counts: Dict[Tuple[int, int], int] = {}
        
        self.is_surge = False
        self.predicted_hotspots: List[Tuple[int, int]] = []
        
    def update(self, current_t: int, new_order_ids: List[int], all_orders: Dict[int, Order]):
        # 1. Ghi nhận số lượng đơn mới xuất hiện
        num_new = len(new_order_ids)
        self.history_counts.append(num_new)
        
        # 2. Lưu lịch sử các điểm pickup kèm timestep
        for oid in new_order_ids:
            if oid in all_orders:
                order = all_orders[oid]
                self.pickup_history.append((current_t, (order.sx, order.sy)))
                
        # Giới hạn window cho pickup history (chỉ giữ 100 steps gần nhất)
        window_limit = 100
        self.pickup_history = [
            (t, pos) for t, pos in self.pickup_history 
            if current_t - t <= window_limit
        ]
        
        # Xây dựng lại pickup_counts từ history với time decay
        self.pickup_counts = {}
        for t, pos in self.pickup_history:
            age = current_t - t
            weight = 0.95 ** age
            self.pickup_counts[pos] = self.pickup_counts.get(pos, 0.0) + weight
                
        # 3. Phát hiện Surge (sliding window 20 steps)
        window_size = min(20, current_t + 1)
        recent_orders = sum(self.history_counts[-window_size:])
        recent_rate = recent_orders / window_size
        
        # Nếu rate sinh đơn gấp > 2.2 lần lambda0 nền -> Surge
        self.is_surge = recent_rate > (self.lambda0 * 2.2)
        
        # 4. Xác định Hotspots (Manhattan distance <= 3)
        if self.pickup_counts:
            candidates = {}
            for (sx, sy), count in self.pickup_counts.items():
                for dr in range(-3, 4):
                    for dc in range(-(3 - abs(dr)), (3 - abs(dr)) + 1):
                        r, c = sx + dr, sy + dc
                        if 0 <= r < self.N and 0 <= c < self.N and self.grid[r][c] == 0:
                            d = abs(dr) + abs(dc)
                            dist_weight = 1.0 / (1.0 + d)
                            candidates[(r, c)] = candidates.get((r, c), 0.0) + count * dist_weight
            
            sorted_candidates = sorted(candidates.items(), key=lambda item: -item[1])
            n_hotspots = min(max(1, self.C // 2), 3)
            self.predicted_hotspots = [pos for pos, score in sorted_candidates[:n_hotspots]]
        else:
            self.predicted_hotspots = []
