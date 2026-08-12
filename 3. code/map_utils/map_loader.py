import os
import glob
import sys
import numpy as np

class MapLoader:
    """
    Utility for scanning, listing, and loading raw BenchNav map datasets (.npz / .pt / .npy) from map_data folder.
    Directly parses raw PyTorch map checkpoints and NumPy compressed maps from masafumiendo/benchnav.
    """
    def __init__(self, map_data_dir="/home/wsly/Nhm2-2.5D/3. code/map_data"):
        self.map_data_dir = map_data_dir
        os.makedirs(self.map_data_dir, exist_ok=True)

    def list_available_maps(self):
        """Scans map_data directory and returns list of raw map filenames (.npz, .pt, and .npy)."""
        pattern_npz = os.path.join(self.map_data_dir, "*.npz")
        pattern_pt = os.path.join(self.map_data_dir, "*.pt")
        pattern_npy = os.path.join(self.map_data_dir, "*.npy")

        map_files = sorted(glob.glob(pattern_npz)) + sorted(glob.glob(pattern_pt)) + sorted(glob.glob(pattern_npy))
        map_list = []
        for f in map_files:
            basename = os.path.basename(f)
            map_name = os.path.splitext(basename)[0]
            map_list.append((map_name, f))
        return map_list

    def select_map_interactive(self):
        """
        Scans map_data folder, prints numbered list of available raw maps,
        and prompts user to choose a map.
        :return: map_name, map_filepath
        """
        map_files = self.list_available_maps()
        if not map_files:
            raise FileNotFoundError(f"No raw map files (.npz, .pt, or .npy) found in {self.map_data_dir}!")

        print("\n==========================================================")
        print("    SELECT A RAW BENCHNAV MAP DATASET FROM map_data/       ")
        print("==========================================================")
        for idx, (map_name, path) in enumerate(map_files, 1):
            ext = os.path.splitext(path)[1]
            print(f"  [{idx}] {map_name:<25} ({os.path.basename(path)})")

        selected_idx = 0
        if len(sys.argv) > 1 and sys.argv[1].isdigit():
            choice = int(sys.argv[1])
            if 1 <= choice <= len(map_files):
                selected_idx = choice - 1
        else:
            try:
                prompt_str = f"\nEnter map number [1-{len(map_files)}] (default 1): "
                user_input = input(prompt_str).strip()
                if user_input.isdigit() and 1 <= int(user_input) <= len(map_files):
                    selected_idx = int(user_input) - 1
            except (EOFError, KeyboardInterrupt):
                selected_idx = 0

        map_name, filepath = map_files[selected_idx]
        print(f">> Selected Map [{selected_idx+1}]: '{map_name}' ({os.path.basename(filepath)})\n")
        return map_name, filepath

    def load_map(self, filepath_or_name):
        """
        Loads raw 2.5D heightmap directly from BenchNav .npz, .pt, or .npy file.
        :return: X, Y, Z, resolution
        """
        if not (filepath_or_name.endswith(".npz") or filepath_or_name.endswith(".pt") or filepath_or_name.endswith(".npy")):
            filepath_npz = os.path.join(self.map_data_dir, f"{filepath_or_name}.npz")
            filepath_pt = os.path.join(self.map_data_dir, f"{filepath_or_name}.pt")
            filepath_npy = os.path.join(self.map_data_dir, f"{filepath_or_name}.npy")
            if os.path.exists(filepath_npz):
                filepath = filepath_npz
            elif os.path.exists(filepath_pt):
                filepath = filepath_pt
            elif os.path.exists(filepath_npy):
                filepath = filepath_npy
            else:
                raise FileNotFoundError(f"Map file not found: {filepath_or_name}")
        else:
            filepath = filepath_or_name

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Map file not found: {filepath}")

        # Parse .npz file
        if filepath.endswith(".npz"):
            data = np.load(filepath, allow_pickle=True)
            if "elevation" in data:
                Z = np.array(data["elevation"])
            elif "Z" in data:
                Z = np.array(data["Z"])
            elif "heights" in data:
                Z = np.array(data["heights"])
            else:
                raise ValueError(f"Unrecognized keys in npz map file: {list(data.keys())}")

            resolution = float(data["resolution"]) if "resolution" in data else 0.25
            ny, nx = Z.shape
            x = np.linspace(0, nx * resolution, nx)
            y = np.linspace(0, ny * resolution, ny)
            X, Y = np.meshgrid(x, y)
            return X, Y, Z, resolution

        # Parse raw PyTorch .pt file
        if filepath.endswith(".pt"):
            import torch
            data = torch.load(filepath, weights_only=False, map_location='cpu')
            tensors = data.get("tensors", {}) if isinstance(data, dict) else {}
            if "heights" in tensors:
                Z = tensors["heights"].cpu().numpy()
            elif isinstance(data, torch.Tensor):
                Z = data.cpu().numpy()
            else:
                raise ValueError(f"Unrecognized PyTorch format in map file: {filepath}")

            resolution = 0.25
            ny, nx = Z.shape
            x = np.linspace(0, nx * resolution, nx)
            y = np.linspace(0, ny * resolution, ny)
            X, Y = np.meshgrid(x, y)
            return X, Y, Z, resolution

        # Parse .npy file
        data = np.load(filepath, allow_pickle=True).item()
        return data["X"], data["Y"], data["Z"], data.get("resolution", 0.25)
