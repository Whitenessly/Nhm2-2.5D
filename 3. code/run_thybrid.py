import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for server environments
import matplotlib.pyplot as plt

from map_utils.map_loader import MapLoader
from map_utils.hybrid_map import HybridMap
from planners.thybrid_a_star import THybridAStar
from benchnav.metrics import PathEvaluator

def main():
    print("==========================================================================")
    print("      TASK 2.1: T-HYBRID A* PLANNER (Liu et al., 2023) - STANDALONE      ")
    print("==========================================================================")

    # 1. Quét list map_data để người dùng chọn map
    loader = MapLoader()
    map_name, filepath = loader.select_map_interactive()

    X, Y, Z, res = loader.load_map(filepath)
    hmap = HybridMap(X, Y, Z, resolution=res)

    # Tự động tính toán điểm Start (góc dưới trái) và Goal (góc trên phải) phù hợp kích thước map
    start_pose = (hmap.width * 0.1, hmap.height * 0.1, 0.0)
    goal_pose = (hmap.width * 0.85, hmap.height * 0.85, 0.0)

    print(f"Executing T-Hybrid A* search on '{map_name}' from {start_pose[:2]} to {goal_pose[:2]}...")
    planner = THybridAStar(hmap, max_iterations=6000)

    t0 = time.time()
    path, success = planner.plan(start_pose, goal_pose)
    t_plan = time.time() - t0

    metrics = PathEvaluator.evaluate_path(path, t_plan, success)
    metrics["map_name"] = map_name
    metrics["algorithm"] = "T-Hybrid A*"

    # 2. In kết quả thực nghiệm
    print("\n==========================================================================")
    print(f" THÀNH CÔNG: KẾT QUẢ THỰC NGHIỆM T-HYBRID A* (Map: {map_name})")
    print("==========================================================================")
    print(f" - Trạng thái thành công: {metrics['success']}")
    print(f" - Thời gian tính toán:   {metrics['planning_time_sec']:.3f} s")
    print(f" - Độ dài đường đi 3D:    {metrics['path_length_3d']:.2f} m")
    print(f" - Độ lệch chuẩn Roll:    {metrics['std_roll_deg']:.2f}° (Max: {metrics['max_roll_deg']:.2f}°)")
    print(f" - Độ lệch chuẩn Pitch:   {metrics['std_pitch_deg']:.2f}° (Max: {metrics['max_pitch_deg']:.2f}°)")
    print(f" - Traversability (tau):  {metrics['mean_traversability']:.3f}")

    # 3. Lưu kết quả JSON & Ảnh minh họa đường đi vào 4. logs
    log_dir = "/home/wsly/Nhm2-2.5D/4. logs"
    os.makedirs(log_dir, exist_ok=True)

    json_path = os.path.join(log_dir, f"thybrid_{map_name}.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    img_path = os.path.join(log_dir, f"thybrid_{map_name}.png")
    fig = plt.figure(figsize=(14, 6))

    # Đồ thị 3D Map và đường đi
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='terrain', alpha=0.7, edgecolor='none')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, label='Độ cao Z (m)')
    if path:
        pts = np.array(path)
        ax1.plot(pts[:, 0], pts[:, 1], pts[:, 2] + 0.1, 'r-', linewidth=3, label='T-Hybrid A* Path')
    ax1.scatter([start_pose[0]], [start_pose[1]], [hmap.get_elevation(start_pose[0], start_pose[1])], color='green', s=120, label='Start')
    ax1.scatter([goal_pose[0]], [goal_pose[1]], [hmap.get_elevation(goal_pose[0], goal_pose[1])], color='gold', marker='*', s=220, label='Goal')
    ax1.set_title(f'T-Hybrid A* 3D Trajectory ({map_name})', fontsize=12, fontweight='bold')
    ax1.legend()

    # Đồ thị 2D Traversability và đường đi
    ax2 = fig.add_subplot(1, 2, 2)
    im = ax2.imshow(hmap.static_traversability, origin='lower', extent=[0, hmap.width, 0, hmap.height], cmap='RdYlGn')
    fig.colorbar(im, ax=ax2, shrink=0.75, label='Traversability tau')
    ax2.contour(X, Y, hmap.occupancy_2d, levels=[0.5], colors='black', linewidths=1.5)
    if path:
        pts = np.array(path)
        ax2.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=2.5, label='T-Hybrid A*')
    ax2.plot(start_pose[0], start_pose[1], 'go', markersize=10, label='Start')
    ax2.plot(goal_pose[0], goal_pose[1], 'y*', markersize=15, label='Goal')
    ax2.set_title(f'2D Map & Traversability ({map_name})', fontsize=12, fontweight='bold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(img_path, dpi=300)
    plt.close()

    print(f"\n[OK] Đã lưu log JSON vào: {json_path}")
    print(f"[OK] Đã lưu ảnh đường đi vào: {img_path}")

if __name__ == "__main__":
    main()
