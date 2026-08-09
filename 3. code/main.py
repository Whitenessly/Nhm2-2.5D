import sys
import os
import matplotlib
matplotlib.use('Agg')  # Headless backend for server environments

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from map_utils.map_loader import MapLoader
from run_thybrid import main as run_thybrid_main
from run_25d_rrt import main as run_25d_rrt_main
from run_benchmark import main as run_benchmark_main

def main():
    print("\n==========================================================================")
    print("      2.5D PATH PLANNING & BENCHNAV SUITE (NHÓM 2)                       ")
    print("==========================================================================")
    print("  [1] Chạy độc lập Thuật toán T-Hybrid A* (Task 2.1 - Liu et al., 2023)")
    print("  [2] Chạy độc lập Thuật toán 2.5D RRT* (Task 2.2 - Steinbauer et al., 2025)")
    print("  [3] Chạy chương trình so sánh BenchNav (Task 2.3 - BenchNav Comparison)")

    selected_mode = 3
    if len(sys.argv) > 1 and sys.argv[1].isdigit():
        choice = int(sys.argv[1])
        if 1 <= choice <= 3:
            selected_mode = choice
    else:
        try:
            choice_str = input("\nChọn chế độ muốn chạy [1-3] (default 3): ").strip()
            if choice_str.isdigit() and 1 <= int(choice_str) <= 3:
                selected_mode = int(choice_str)
        except (EOFError, KeyboardInterrupt):
            selected_mode = 3

    print(f"\n>> Đã chọn Chế độ [{selected_mode}]. Quét danh sách bản đồ trong map_data/...\n")

    if selected_mode == 1:
        run_thybrid_main()
    elif selected_mode == 2:
        run_25d_rrt_main()
    elif selected_mode == 3:
        run_benchmark_main()

if __name__ == "__main__":
    main()
