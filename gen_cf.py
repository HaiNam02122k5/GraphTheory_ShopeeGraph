import os
import random

# ==========================================
# 5 Cấp độ Quy mô Cơ sở (Base Scales)
# ==========================================
BASE_SCALES = {
    "default":  {"N": 15, "C": 3, "G": 50,  "T": 600,  "K": 3, "W": 20.0},
    "medium":   {"N": 30, "C": 6, "G": 150, "T": 1200, "K": 3, "W": 20.0},
    "balanced": {"N": 50, "C": 12, "G": 400, "T": 1800, "K": 3, "W": 20.0},
    "large":    {"N": 80, "C": 18, "G": 900, "T": 2400, "K": 3, "W": 20.0},
    "max":      {"N": 100, "C": 25, "G": 1500, "T": 2400, "K": 3, "W": 20.0}
}

# ==========================================
# Các hàm sinh bản đồ chuyên biệt & kiểm tra liên thông
# ==========================================

def check_connectivity(grid):
    N = len(grid)
    start = None
    free_cells_count = 0
    for r in range(N):
        for c in range(N):
            if grid[r][c] == 0:
                free_cells_count += 1
                if start is None:
                    start = (r, c)
    if start is None:
        return False
        
    visited = {start}
    queue = [start]
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < N and 0 <= nc < N and grid[nr][nc] == 0:
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))
    return len(visited) == free_cells_count

def generate_connected_random_map(N, obstacle_density=0.15):
    """Sinh bản đồ ngẫu nhiên đô thị lưới mở, đảm bảo liên thông hoàn toàn."""
    for _ in range(2000):
        grid = [[0 for _ in range(N)] for _ in range(N)]
        for i in range(N):
            for j in range(N):
                if i == 0 or i == N - 1 or j == 0 or j == N - 1:
                    grid[i][j] = 1
                else:
                    if random.random() < obstacle_density:
                        grid[i][j] = 1
        if check_connectivity(grid):
            return grid
            
    # Dự phòng: Nếu sau 2000 lần sinh ngẫu nhiên vẫn bị cô lập (hiếm), trả về Open Field
    return generate_open_field(N)

def generate_open_field(N):
    """Bình nguyên mở rộng (V1), không vật cản bên trong, bao tường bên ngoài."""
    grid = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        grid[i][0] = grid[i][N-1] = grid[0][i] = grid[N-1][i] = 1
    return grid

def generate_multilane_map(N, v_level):
    """
    Sinh các đường đi mô phỏng trục đường chính với số làn đường:
    - V1: 1 làn
    - V2: mix 1 và 2 làn
    - V3: 2 làn
    - V4: mix 2 và 3 làn
    - V5: 3 làn
    """
    grid = [[1 for _ in range(N)] for _ in range(N)]
    
    if v_level == 1:
        widths = [1]
        spacing = 3
    elif v_level == 2:
        widths = [1, 2]
        spacing = 4
    elif v_level == 3:
        widths = [2]
        spacing = 4
    elif v_level == 4:
        widths = [2, 3]
        spacing = 5
    else:  # v_level == 5
        widths = [3]
        spacing = 5

    roads_x = []
    curr = 1
    idx = 0
    while curr < N - 1:
        w = widths[idx % len(widths)]
        if curr + w > N - 1:
            w = (N - 1) - curr
            if w <= 0:
                break
        roads_x.append((curr, curr + w))
        curr += w + spacing
        idx += 1

    roads_y = []
    curr = 1
    idx = 0
    while curr < N - 1:
        w = widths[idx % len(widths)]
        if curr + w > N - 1:
            w = (N - 1) - curr
            if w <= 0:
                break
        roads_y.append((curr, curr + w))
        curr += w + spacing
        idx += 1

    for r in range(1, N - 1):
        for c in range(1, N - 1):
            in_road_r = any(start <= r < end for start, end in roads_x)
            in_road_c = any(start <= c < end for start, end in roads_y)
            if in_road_r or in_road_c:
                grid[r][c] = 0

    # Bao tường ngoài cùng
    for i in range(N):
        grid[i][0] = grid[i][N-1] = grid[0][i] = grid[N-1][i] = 1
        
    return grid

def generate_ring_road(N):
    """Hệ thống đại lộ vành đai đồng tâm chạy quanh lõi."""
    grid = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        grid[i][0] = grid[i][N-1] = grid[0][i] = grid[N-1][i] = 1
        
    center = N // 2
    # Vẽ các bức tường vuông đồng tâm ngăn cách các vành đai
    for d in range(3, center - 1, 4):
        for i in range(d, N - d):
            grid[d][i] = 1
            grid[N - 1 - d][i] = 1
            grid[i][d] = 1
            grid[i][N - 1 - d] = 1
            
    # Tạo các cổng liên thông (đường xuyên tâm chữ thập)
    for i in range(1, N - 1):
        grid[center][i] = 0
        grid[i][center] = 0
        
    return grid

def generate_maze(N):
    """Mạng lưới bẫy đường cụt hiểm trở (Maze + đục lỗ ngẫu nhiên)."""
    grid = [[1 for _ in range(N)] for _ in range(N)]
    
    stack = [(1, 1)]
    grid[1][1] = 0
    
    while stack:
        r, c = stack[-1]
        neighbors = []
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 0 < nr < N - 1 and 0 < nc < N - 1 and grid[nr][nc] == 1:
                neighbors.append((nr, nc))
        if neighbors:
            nr, nc = random.choice(neighbors)
            grid[(r + nr) // 2][(c + nc) // 2] = 0
            grid[nr][nc] = 0
            stack.append((nr, nc))
        else:
            stack.pop()
            
    # Mở thêm 20% tường ngẫu nhiên để tạo các vòng tuần hoàn nhưng vẫn nhiều hẻm cụt
    for r in range(1, N - 1):
        for c in range(1, N - 1):
            if grid[r][c] == 1 and random.random() < 0.20:
                grid[r][c] = 0
                
    if not check_connectivity(grid):
        return generate_multilane_map(N, v_level=1)
    return grid


# ==========================================
# Ghi cấu hình ra file
# ==========================================

def format_map_to_string(grid):
    return "\n".join(" ".join(map(str, row)) for row in grid)

def write_config_to_file(filepath, configs, scenario_name):
    """Ghi 5 cấu hình của một quy mô vào file tương ứng."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("# =============================================================\n")
        f.write(f"# MAPD Auto-generated test configs: Scenario {scenario_name}\n")
        f.write("# =============================================================\n\n")
        f.write("[SEED]\n")
        f.write("base_seed = 42\n\n")
        
        for cfg in configs:
            f.write("[CONFIG]\n")
            f.write(f"name    = {cfg['name']}\n")
            f.write(f"N       = {cfg['N']}\n")
            f.write(f"C       = {cfg['C']}\n")
            f.write(f"G       = {cfg['G']}\n")
            f.write(f"T       = {cfg['T']}\n")
            
            k_max_str = " ".join(map(str, cfg['K_max']))
            f.write(f"K_max   = {k_max_str}\n")
            
            w_max_str = " ".join(f"{w:.1f}" for w in cfg['W_max'])
            f.write(f"W_max   = {w_max_str}\n")
            
            # Các thông số phụ cho kịch bản advanced
            if 'surge_windows' in cfg:
                sw_str = " ".join(f"{ts} {te}" for ts, te in cfg['surge_windows'])
                f.write(f"surge_windows = {sw_str}\n")
            if 'hotspots' in cfg:
                hs_str = " ".join(f"{r} {c}" for r, c in cfg['hotspots'])
                f.write(f"hotspots = {hs_str}\n")
            if 'surge_amplitude' in cfg:
                f.write(f"surge_amplitude = {cfg['surge_amplitude']:.1f}\n")
                
            f.write("[MAP]\n")
            f.write(format_map_to_string(cfg['map']) + "\n")
            f.write("[END]\n\n")


# ==========================================
# Trình điều khiển chính để sinh tất cả các kịch bản
# ==========================================

def get_free_cells(grid):
    return [(r, c) for r, row in enumerate(grid) for c, val in enumerate(row) if val == 0]

def main():
    random.seed(42)
    os.makedirs("./test_config", exist_ok=True)
    
    scenarios = ["balanced", "multilane", "weight", "shipper", "deadline", "topology", "advanced", "adaptability"]
    
    for scenario in scenarios:
        scenario_dir = os.path.join("./test_config", scenario)
        os.makedirs(scenario_dir, exist_ok=True)
        
        # Với mỗi kịch bản, ta sinh 5 file cơ sở tương ứng với 5 quy mô (base scales)
        for scale_name, base in BASE_SCALES.items():
            configs = []
            
            # Sinh chung 1 bản đồ cơ sở cho các kịch bản không đổi bản đồ dọc theo V1->V5
            # Bản đồ ngẫu nhiên liên thông tiêu chuẩn có 15% cản
            base_grid = generate_connected_random_map(base["N"], obstacle_density=0.15)
            
            # Sinh 5 phiên bản (V1 đến V5)
            for v in range(1, 6):
                cfg = {
                    "name": f"{scenario}_{scale_name}_v{v}",
                    "N": base["N"],
                    "C": base["C"],
                    "G": base["G"],
                    "T": base["T"],
                    "K_max": [base["K"]] * base["C"],
                    "W_max": [base["W"]] * base["C"],
                    "map": base_grid
                }
                
                # --------------------------------------------------
                # Kịch bản 1: balanced (Độ chịu tải đơn hàng)
                # --------------------------------------------------
                if scenario == "balanced":
                    # Biến thiên số lượng đơn G từ 40% đến 170%
                    rates = [0.4, 0.7, 1.0, 1.35, 1.7]
                    cfg["G"] = int(base["G"] * rates[v - 1])
                    cfg["map"] = generate_connected_random_map(base["N"], obstacle_density=0.10) # lưới mở ít vật cản
                    
                # --------------------------------------------------
                # Kịch bản 2: multilane (Độ rộng làn đường)
                # --------------------------------------------------
                elif scenario == "multilane":
                    # Thay đổi bản đồ ứng với từng V
                    cfg["map"] = generate_multilane_map(base["N"], v_level=v)
                    
                # --------------------------------------------------
                # Kịch bản 3: weight (Ràng buộc tải trọng shipper)
                # --------------------------------------------------
                elif scenario == "weight":
                    # Biến thiên K_max và W_max từ cực thấp đến vô hạn
                    k_vals = [1, 2, 3, 4, 5]
                    w_vals = [4.0, 15.0, 30.0, 60.0, 120.0]
                    cfg["K_max"] = [k_vals[v - 1]] * base["C"]
                    cfg["W_max"] = [w_vals[v - 1]] * base["C"]
                    
                # --------------------------------------------------
                # Kịch bản 4: shipper (Số lượng Shipper C)
                # --------------------------------------------------
                elif scenario == "shipper":
                    # Biến thiên C, khống chế tối đa 25 theo ràng buộc hệ thống
                    if scale_name == "default":
                        c_vals = [1, 2, 3, 4, 5]
                    elif scale_name == "medium":
                        c_vals = [1, 3, 6, 7, 8]
                    elif scale_name == "balanced":
                        c_vals = [2, 6, 12, 14, 16]
                    elif scale_name == "large":
                        c_vals = [3, 9, 18, 20, 22]
                    else: # max
                        c_vals = [5, 12, 25, 25, 25] # max limit is 25
                    
                    c_val = c_vals[v - 1]
                    cfg["C"] = c_val
                    cfg["K_max"] = [base["K"]] * c_val
                    cfg["W_max"] = [base["W"]] * c_val
                    
                # --------------------------------------------------
                # Kịch bản 5: deadline (Áp lực thời gian T)
                # --------------------------------------------------
                elif scenario == "deadline":
                    # Biến thiên T, khống chế tối đa 2400
                    if scale_name == "default":
                        t_vals = [180, 360, 600, 900, 1200]
                    elif scale_name == "medium":
                        t_vals = [360, 720, 1200, 1800, 2400]
                    elif scale_name == "balanced":
                        t_vals = [540, 1080, 1800, 2400, 2400]
                    elif scale_name == "large":
                        t_vals = [720, 1440, 2400, 2400, 2400]
                    else: # max
                        t_vals = [720, 1440, 2400, 2400, 2400]
                    
                    cfg["T"] = t_vals[v - 1]
                    
                # --------------------------------------------------
                # Kịch bản 6: topology (Kiến trúc bản đồ)
                # --------------------------------------------------
                elif scenario == "topology":
                    if v == 1:
                        cfg["map"] = generate_open_field(base["N"])
                    elif v == 2:
                        cfg["map"] = generate_multilane_map(base["N"], v_level=1)
                    elif v == 3:
                        cfg["map"] = generate_ring_road(base["N"])
                    elif v == 4:
                        cfg["map"] = generate_multilane_map(base["N"], v_level=4)
                    else: # 5
                        cfg["map"] = generate_maze(base["N"])
                        
                # --------------------------------------------------
                # Kịch bản 7: advanced (Nâng cao Phase 2 - Surge/Hotspot)
                # --------------------------------------------------
                elif scenario == "advanced":
                    # Tăng dần surge_amplitude từ 1.0 đến 6.0
                    amps = [1.0, 2.2, 3.5, 4.8, 6.0]
                    cfg["surge_amplitude"] = amps[v - 1]
                    
                    # 2 khung giờ surge cố định dựa trên thời gian T
                    t_max = base["T"]
                    cfg["surge_windows"] = [
                        (int(t_max * 0.2), int(t_max * 0.4)),
                        (int(t_max * 0.6), int(t_max * 0.8))
                    ]
                    
                    # Lấy ngẫu nhiên 3 ô trống làm hotspot
                    free_cells = get_free_cells(base_grid)
                    cfg["hotspots"] = random.sample(free_cells, min(3, len(free_cells)))
                    
                # --------------------------------------------------
                # Kịch bản 8: adaptability (Độ thích nghi - Ngẫu nhiên hóa)
                # --------------------------------------------------
                elif scenario == "adaptability":
                    # Ngẫu nhiên hóa N theo scale
                    n_min, n_max = {
                        "default": (12, 18),
                        "medium": (25, 35),
                        "balanced": (40, 60),
                        "large": (70, 80), # 80 is baseline max for large
                        "max": (90, 100)
                    }[scale_name]
                    n_rand = random.randint(n_min, n_max)
                    cfg["N"] = n_rand
                    
                    # Ngẫu nhiên hóa C (C <= 25)
                    c_min, c_max = {
                        "default": (1, 5),
                        "medium": (2, 8),
                        "balanced": (4, 16),
                        "large": (6, 22),
                        "max": (10, 25)
                    }[scale_name]
                    c_rand = min(25, random.randint(c_min, c_max))
                    cfg["C"] = c_rand
                    
                    # Ngẫu nhiên hóa G & T (T <= 2400)
                    g_rand = int(base["G"] * random.uniform(0.6, 1.4))
                    t_rand = min(2400, int(base["T"] * random.uniform(0.6, 1.4)))
                    cfg["G"] = g_rand
                    cfg["T"] = t_rand
                    
                    # Ngẫu nhiên giới hạn vật lý của shipper độc lập từng tác tử
                    k_choices = [1, 2, 3, 4, 5]
                    w_choices = [4.0, 10.0, 20.0, 30.0, 60.0, 80.0, 120.0]
                    cfg["K_max"] = [random.choice(k_choices) for _ in range(c_rand)]
                    cfg["W_max"] = [random.choice(w_choices) for _ in range(c_rand)]
                    
                    # Ngẫu nhiên lựa chọn địa hình
                    topo_choice = random.choice(["open", "grid", "ring", "mixed", "maze", "random"])
                    if topo_choice == "open":
                        cfg["map"] = generate_open_field(n_rand)
                    elif topo_choice == "grid":
                        cfg["map"] = generate_multilane_map(n_rand, v_level=1)
                    elif topo_choice == "ring":
                        cfg["map"] = generate_ring_road(n_rand)
                    elif topo_choice == "mixed":
                        cfg["map"] = generate_multilane_map(n_rand, v_level=random.choice([2, 4]))
                    elif topo_choice == "maze":
                        cfg["map"] = generate_maze(n_rand)
                    else:  # random
                        cfg["map"] = generate_connected_random_map(n_rand, obstacle_density=random.uniform(0.05, 0.25))
                        
                    # Có 50% cơ hội xuất hiện surge & hotspots
                    if random.random() < 0.5:
                        cfg["surge_amplitude"] = round(random.uniform(1.0, 6.0), 1)
                        cfg["surge_windows"] = [
                            (int(t_rand * 0.15), int(t_rand * 0.35)),
                            (int(t_rand * 0.55), int(t_rand * 0.75))
                        ]
                        free_cells = get_free_cells(cfg["map"])
                        cfg["hotspots"] = random.sample(free_cells, min(random.randint(1, 4), len(free_cells)))
                        
                configs.append(cfg)
                
            # Lưu 5 cấu hình của scale_name vào file
            filename = f"{scale_name}.txt"
            filepath = os.path.join(scenario_dir, filename)
            write_config_to_file(filepath, configs, scenario)
            
    print("Hoàn tất sinh toàn bộ 40 file cấu hình kiểm thử thành công.")

if __name__ == "__main__":
    main()