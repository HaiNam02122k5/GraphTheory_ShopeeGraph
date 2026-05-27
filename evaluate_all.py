import os
import sys
import time
import json
import copy
import hashlib
from collections import deque
from typing import Any, Dict, List

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKSPACE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, WORKSPACE_DIR)
sys.path.insert(0, os.path.join(WORKSPACE_DIR, "solvers"))

from env import DeliveryEnv, load_config
import importlib.util

def load_solver_class(class_name: str, file_name: str):
    path = os.path.join(WORKSPACE_DIR, "solvers", file_name)
    spec = importlib.util.spec_from_file_location(class_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, class_name)

def _stable_config_seed(config_name: str, base_seed: int) -> int:
    digest = hashlib.md5(f"{base_seed}:{config_name}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)

def _run_solver(solver_cls: Any, cfg: dict, seed: int) -> dict:
    env_cfg = copy.deepcopy(cfg)
    env = DeliveryEnv(env_cfg, seed=seed)
    solver = solver_cls(env)
    return solver.run()

def main():
    print("Loading solver classes...")
    VRP_cls = load_solver_class("VRPOrToolsSolver", "vrp_ortools.py")
    BFS_cls = load_solver_class("GreedyBFS", "greedy_bfs.py")
    
    solvers = {
        "VRPOrToolsSolver": VRP_cls,
        "GreedyBFS": BFS_cls
    }
    
    scenarios = ["shipper", "adaptability", "weight", "balanced", "topology", "deadline", "multilane", "advanced"]
    config_types = ["large"]
    
    all_results = []
    
    # Process each scenario and config type
    total_configs = len(scenarios) * len(config_types)
    current_count = 0
    
    print(f"Starting evaluations for {total_configs} configuration files...")
    
    for scenario in scenarios:
        for cfg_type in config_types:
            current_count += 1
            config_file = f"{cfg_type}.txt"
            config_path = os.path.join(WORKSPACE_DIR, "test_config", scenario, config_file)
            
            if not os.path.exists(config_path):
                print(f"[{current_count}/{total_configs}] Config not found: {config_path}")
                continue
                
            print(f"[{current_count}/{total_configs}] Running {scenario}/{config_file}...")
            configs = load_config(config_path)
            
            for cfg in configs:
                name = cfg.get("name", "unknown")
                config_seed = _stable_config_seed(str(name), cfg['base_seed'])
                
                for solver_name, solver_cls in solvers.items():
                    start_time = time.time()
                    try:
                        res = _run_solver(solver_cls, cfg, config_seed)
                        net_reward = res.get("net_reward", 0.0)
                        delivered = res.get("delivered", 0)
                        total_orders = res.get("total_orders", cfg["G"])
                        on_time = res.get("on_time", 0)
                        
                        delivery_rate = (delivered / total_orders * 100) if total_orders > 0 else 0.0
                        on_time_rate = (on_time / delivered * 100) if delivered > 0 else 0.0
                    except Exception as e:
                        print(f"  Error running {solver_name} on {name}: {e}")
                        net_reward = 0.0
                        delivery_rate = 0.0
                        on_time_rate = 0.0
                        delivered = 0
                        total_orders = cfg["G"]
                        on_time = 0
                        
                    elapsed = time.time() - start_time
                    
                    all_results.append({
                        "scenario": scenario,
                        "config_type": cfg_type,
                        "config_name": name,
                        "solver": solver_name,
                        "net_reward": round(net_reward, 2),
                        "delivery_rate": round(delivery_rate, 1),
                        "on_time_rate": round(on_time_rate, 1),
                        "delivered": delivered,
                        "total_orders": total_orders,
                        "on_time": on_time,
                        "time_sec": round(elapsed, 2)
                    })
    
    # Save raw results to JSON
    out_dir = os.path.join(WORKSPACE_DIR, "results")
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "evaluation_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved raw results to {json_path}")
    
    # Generate Markdown Summary
    md_path = os.path.join(out_dir, "evaluation_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Báo cáo kết quả thử nghiệm\n\n")
        f.write("So sánh hiệu năng giữa `VRPOrToolsSolver_t` và `GreedyBFS_t` trên các kịch bản và cấu hình khác nhau.\n\n")
        
        # Scenario Aggregated Table
        f.write("## 1. Tổng hợp theo Kịch bản & Loại cấu hình (Trung bình cộng)\n\n")
        f.write("| Kịch bản | Cấu hình | Solver | Tổng điểm | Net Reward trung bình | % Giao trung bình | % Đúng hạn trung bình | Thời gian (s) trung bình |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        
        # Group and average
        aggregated = {}
        for r in all_results:
            key = (r["scenario"], r["config_type"], r["solver"])
            if key not in aggregated:
                aggregated[key] = {"net_reward": [], "delivery_rate": [], "on_time_rate": [], "time_sec": []}
            aggregated[key]["net_reward"].append(r["net_reward"])
            aggregated[key]["delivery_rate"].append(r["delivery_rate"])
            aggregated[key]["on_time_rate"].append(r["on_time_rate"])
            aggregated[key]["time_sec"].append(r["time_sec"])
            
        for key in sorted(aggregated.keys()):
            scen, cfg_t, solv = key
            vals = aggregated[key]
            sum_reward = sum(vals["net_reward"])
            avg_reward = sum_reward / len(vals["net_reward"])
            avg_deliv = sum(vals["delivery_rate"]) / len(vals["delivery_rate"])
            avg_ontime = sum(vals["on_time_rate"]) / len(vals["on_time_rate"])
            avg_time = sum(vals["time_sec"]) / len(vals["time_sec"])
            f.write(f"| {scen} | {cfg_t} | `{solv}` | {sum_reward:.2f} | {avg_reward:.2f} | {avg_deliv:.1f}% | {avg_ontime:.1f}% | {avg_time:.2f}s |\n")
            
        # Overall Summary Table
        f.write("\n## 2. Tổng điểm toàn bộ các kịch bản\n\n")
        f.write("| Solver | Tổng điểm | Tổng Net Reward | Thời gian chạy tổng cộng (s) |\n")
        f.write("|---|---|---|---|\n")
        
        solvers_totals = {}
        for r in all_results:
            solv = r["solver"]
            if solv not in solvers_totals:
                solvers_totals[solv] = {"net_reward": 0.0, "time_sec": 0.0}
            solvers_totals[solv]["net_reward"] += r["net_reward"]
            solvers_totals[solv]["time_sec"] += r["time_sec"]
            
        for solv, totals in solvers_totals.items():
            f.write(f"| `{solv}` | {totals['net_reward']:.2f} | {totals['net_reward']:.2f} | {totals['time_sec']:.2f}s |\n")
            
        # Detailed Table
        f.write("\n## 3. Chi tiết từng cấu hình con\n\n")
        f.write("| Kịch bản | Loại | Config Name | Solver | Net Reward | Giao/Tổng | Đúng hạn | Thời gian (s) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in all_results:
            f.write(f"| {r['scenario']} | {r['config_type']} | {r['config_name']} | `{r['solver']}` | {r['net_reward']} | {r['delivered']}/{r['total_orders']} | {r['on_time']} | {r['time_sec']}s |\n")
            
    print(f"Saved markdown summary to {md_path}")

if __name__ == "__main__":
    main()
