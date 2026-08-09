import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for server environments
import matplotlib.pyplot as plt

from map_utils.map_loader import MapLoader
from map_utils.hybrid_map import HybridMap
from map_utils.point_picker import select_start_goal
from planners.thybrid_a_star import THybridAStar
from planners.rrt_star_2_5d import RRTStar25D
from planners.baselines import BaselineAStar, BaselineRRTStar
from benchnav.metrics import PathEvaluator

def main():
    print("==========================================================================")
    print("      TASK 2.3: BENCHNAV COMPARATIVE ANALYSIS SUITE - STANDALONE          ")
    print("==========================================================================")

    # 1. Quét list map_data để người dùng chọn map
    loader = MapLoader()
    map_name, filepath = loader.select_map_interactive()

    X, Y, Z, res = loader.load_map(filepath)
    hmap = HybridMap(X, Y, Z, resolution=res)

    # 2. Chọn vị trí Start & Goal bằng click trỏ chuột trên popup (hoặc nhập tọa độ)
    start_pose, goal_pose = select_start_goal(hmap, map_name)

    print(f"Executing BenchNav benchmark on '{map_name}' from {start_pose[:2]} to {goal_pose[:2]}...\n")

    planners = {
        "Baseline 2D A*": BaselineAStar(hmap),
        "Baseline 2D RRT*": BaselineRRTStar(hmap, max_iterations=1000),
        "T-Hybrid A* (Task 2.1)": THybridAStar(hmap, max_iterations=4000),
        "2.5D RRT* (Task 2.2)": RRTStar25D(hmap, max_iterations=1200)
    }

    results = []
    paths_dict = {}

    for name, planner in planners.items():
        t0 = time.time()
        path, success = planner.plan(start_pose, goal_pose)
        t_plan = time.time() - t0

        metrics = PathEvaluator.evaluate_path(path, t_plan, success)
        metrics["map_name"] = map_name
        metrics["algorithm"] = name
        results.append(metrics)
        paths_dict[name] = path

        print(f"  [{name:<22}] -> Success: {str(success):<5} | Time: {t_plan:.3f}s | "
              f"3D Length: {metrics['path_length_3d']:.2f}m | "
              f"Std Roll: {metrics['std_roll_deg']:.2f}° | "
              f"Mean Tau: {metrics['mean_traversability']:.3f}")

    # 3. In bảng tổng hợp so sánh BenchNav
    print("\n==========================================================================")
    print(f" BẢNG SO SÁNH THUẬT TOÁN BENCHNAV (Map: {map_name})")
    print("==========================================================================")
    print(f"{'Algorithm':<25} | {'Success':<8} | {'Time (s)':<10} | {'3D Length':<10} | {'Std Roll':<10} | {'Mean Tau':<10}")
    print("-" * 85)
    for r in results:
        print(f"{r['algorithm']:<25} | {str(r['success']):<8} | {r['planning_time_sec']:<10.3f} | {r['path_length_3d']:<10.2f} | {r['std_roll_deg']:<10.2f} | {r['mean_traversability']:<10.3f}")

    # 4. Lưu log JSON & Ảnh so sánh với CHÚ THÍCH ĐẶT NGOÀI KHUNG BẢN ĐỒ
    log_dir = "/home/wsly/Nhm2-2.5D/4. logs"
    os.makedirs(log_dir, exist_ok=True)

    json_path = os.path.join(log_dir, f"benchnav_{map_name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    img_path = os.path.join(log_dir, f"benchnav_{map_name}.png")
    fig = plt.figure(figsize=(16, 6))

    # Đồ thị 3D Surface với tất cả các thuật toán
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='terrain', alpha=0.6, edgecolor='none')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Độ cao Z (m)')

    colors = {'Baseline 2D A*': 'gray', 'Baseline 2D RRT*': 'orange', 'T-Hybrid A* (Task 2.1)': 'red', '2.5D RRT* (Task 2.2)': 'blue'}
    styles = {'Baseline 2D A*': ':', 'Baseline 2D RRT*': '-.', 'T-Hybrid A* (Task 2.1)': '-', '2.5D RRT* (Task 2.2)': '--'}

    for name, path in paths_dict.items():
        if path:
            pts = np.array(path)
            ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2] + 0.1, color=colors[name], linestyle=styles[name], linewidth=2.5, label=name)

    ax1.scatter([start_pose[0]], [start_pose[1]], [hmap.get_elevation(start_pose[0], start_pose[1])], color='green', s=120, label='Start')
    ax1.scatter([goal_pose[0]], [goal_pose[1]], [hmap.get_elevation(goal_pose[0], goal_pose[1])], color='gold', marker='*', s=220, label='Goal')
    ax1.set_title(f'BenchNav 3D Path Comparison ({map_name})', fontsize=12, fontweight='bold')

    # Đồ thị 2D Traversability với tất cả các đường đi
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(hmap.static_traversability, origin='lower', extent=[0, hmap.width, 0, hmap.height], cmap='RdYlGn')
    fig.colorbar(im, ax=ax2, shrink=0.75, label='Traversability tau')
    ax2.contour(X, Y, hmap.occupancy_2d, levels=[0.5], colors='black', linewidths=1.5)

    for name, path in paths_dict.items():
        if path:
            pts = np.array(path)
            ax2.plot(pts[:, 0], pts[:, 1], color=colors[name], linestyle=styles[name], linewidth=2.5, label=name)

    ax2.plot(start_pose[0], start_pose[1], 'go', markersize=10, label='Start')
    ax2.plot(goal_pose[0], goal_pose[1], 'y*', markersize=15, label='Goal')
    ax2.set_title(f'2D Map & Traversability Comparison ({map_name})', fontsize=12, fontweight='bold')

    # ĐẶT CHÚ THÍCH (LEGEND) NGOÀI KHUNG BẢN ĐỒ BÊN PHẢI ĐỂ KHÔNG BỊ ĐÈ LÊN NỘI DUNG
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, bbox_to_anchor=(1.28, 1.0), loc='upper left', borderaxespad=0., fontsize=9.5, frameon=True)

    plt.tight_layout()
    plt.savefig(img_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n[OK] Đã lưu log JSON vào: {json_path}")
    print(f"[OK] Đã lưu ảnh so sánh đường đi vào: {img_path}")

if __name__ == "__main__":
    main()
