import os
import sys
import argparse
import matplotlib
matplotlib.use('Agg')

from map_utils.map_loader import MapLoader
from run_thybrid import run_thybrid_standalone
from run_25d_rrt import run_25d_rrt_standalone
from run_benchmark import run_benchnav_map

def interactive_menu(map_selection=None, algo_selection=None):
    """
    Interactive Terminal UI for selecting 2.5D Path Planning Algorithms and Heightmap Datasets.
    """
    print("\n==========================================================================")
    print("      2.5D PATH PLANNING & BENCHNAV INTERACTIVE SUITE (GROUP 2)           ")
    print("==========================================================================")

    loader = MapLoader()
    map_files = loader.list_available_maps()

    if not map_files:
        print("No map files found in map_data! Generating default dataset...")
        from generate_map_dataset import generate_default_map_dataset
        generate_default_map_dataset()
        map_files = loader.list_available_maps()

    print("\n--- STEP 1: Select Heightmap Dataset from map_data/ ---")
    for idx, (map_name, path) in enumerate(map_files, 1):
        print(f"  [{idx}] {map_name:<22} ({os.path.basename(path)})")

    selected_map_idx = 0
    if map_selection is not None:
        if isinstance(map_selection, int) and 1 <= map_selection <= len(map_files):
            selected_map_idx = map_selection - 1
        elif isinstance(map_selection, str):
            for i, (mname, _) in enumerate(map_files):
                if mname == map_selection:
                    selected_map_idx = i
                    break
    else:
        try:
            choice_str = input(f"\nSelect map number [1-{len(map_files)}] (default 1): ").strip()
            if choice_str.isdigit() and 1 <= int(choice_str) <= len(map_files):
                selected_map_idx = int(choice_str) - 1
        except (EOFError, KeyboardInterrupt):
            selected_map_idx = 0

    selected_map_name = map_files[selected_map_idx][0]
    print(f">> Selected Map: '{selected_map_name}'")

    print("\n--- STEP 2: Select Algorithm / Mode ---")
    print("  [1] T-Hybrid A*")
    print("  [2] 2.5D RRT*")
    print("  [3] BenchNav Comparative Analysis (T-Hybrid A* vs 2.5D RRT*)")

    selected_algo_idx = 3
    if algo_selection is not None and 1 <= algo_selection <= 3:
        selected_algo_idx = algo_selection
    else:
        try:
            choice_str = input("\nSelect algorithm number [1-3] (default 3): ").strip()
            if choice_str.isdigit() and 1 <= int(choice_str) <= 3:
                selected_algo_idx = int(choice_str)
        except (EOFError, KeyboardInterrupt):
            selected_algo_idx = 3

    print(f"\n>> Executing Mode [{selected_algo_idx}] on Map: '{selected_map_name}'...\n")

    if selected_algo_idx == 1:
        run_thybrid_standalone(selected_map_name)
    elif selected_algo_idx == 2:
        run_25d_rrt_standalone(selected_map_name)
    elif selected_algo_idx == 3:
        run_benchnav_map(selected_map_name, num_runs=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive UI for selecting map and algorithm")
    parser.add_argument("--map", type=str, default=None, help="Map name or 1-based index")
    parser.add_argument("--algo", type=int, default=None, help="Algorithm choice [1-5]")
    args = parser.parse_args()

    map_val = int(args.map) if (args.map and args.map.isdigit()) else args.map
    interactive_menu(map_val, args.algo)

