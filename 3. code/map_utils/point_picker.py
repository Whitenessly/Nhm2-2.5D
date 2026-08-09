import os
import sys
import numpy as np

def select_start_goal(hmap, map_name="map"):
    """
    Pops up an interactive Matplotlib GUI window allowing the user to click
    on the 2D map to set Start (Click 1) and Goal (Click 2) positions.
    Falls back gracefully to default poses / CLI input if GUI/DISPLAY is unavailable.
    """
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    if has_display:
        try:
            import matplotlib
            # Check if backend supports interactive GUI
            if matplotlib.get_backend().lower() not in ['agg', 'template']:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(8, 8))
                im = ax.imshow(hmap.static_traversability, origin='lower',
                               extent=[0, hmap.width, 0, hmap.height], cmap='RdYlGn')
                fig.colorbar(im, ax=ax, label='Traversability tau')
                ax.contour(hmap.X, hmap.Y, hmap.occupancy_2d, levels=[0.5], colors='black', linewidths=1.2)

                ax.set_title(f"[{map_name}] CLICK 1: Select START (Green) | CLICK 2: Select GOAL (Gold)",
                             fontsize=11, fontweight='bold', color='darkblue')
                ax.set_xlabel("X (meters)")
                ax.set_ylabel("Y (meters)")
                ax.grid(True, linestyle='--', alpha=0.5)

                print("\n>> POP-UP WINDOW OPENED: Click 2 points on the Map:")
                print("   - Click 1: START Point")
                print("   - Click 2: GOAL Point")

                pts = plt.ginput(2, timeout=30)
                plt.close(fig)

                if len(pts) == 2:
                    (sx, sy), (gx, gy) = pts[0], pts[1]
                    sx, sy = max(0.5, min(hmap.width - 0.5, sx)), max(0.5, min(hmap.height - 0.5, sy))
                    gx, gy = max(0.5, min(hmap.width - 0.5, gx)), max(0.5, min(hmap.height - 0.5, gy))

                    syaw = np.arctan2(gy - sy, gx - sx)
                    gyaw = syaw
                    print(f">> Mouse Selected START: ({sx:.2f}, {sy:.2f}) | GOAL: ({gx:.2f}, {gy:.2f})")
                    return (sx, sy, syaw), (gx, gy, gyaw)
        except Exception as e:
            print(f">> GUI Window Notice: {e}. Falling back to default selection mode...")

    # Headless / Default fallback mode
    default_start = (hmap.width * 0.1, hmap.height * 0.1, 0.0)
    default_goal = (hmap.width * 0.85, hmap.height * 0.85, 0.0)

    print("\n----------------------------------------------------------")
    print(f" Start & Goal Poses for Map '{map_name}':")
    print(f"  - START: ({default_start[0]:.2f}, {default_start[1]:.2f})")
    print(f"  - GOAL:  ({default_goal[0]:.2f}, {default_goal[1]:.2f})")
    print("----------------------------------------------------------")

    try:
        ans = input("Press ENTER to use defaults, or type 'c' for custom (x,y): ").strip().lower()
        if ans == 'c':
            s_str = input(f"Enter START x,y (e.g. {default_start[0]:.1f},{default_start[1]:.1f}): ").strip()
            g_str = input(f"Enter GOAL x,y (e.g. {default_goal[0]:.1f},{default_goal[1]:.1f}): ").strip()

            sx, sy = map(float, s_str.split(','))
            gx, gy = map(float, g_str.split(','))
            syaw = np.arctan2(gy - sy, gx - sx)
            return (sx, sy, syaw), (gx, gy, syaw)
    except (EOFError, KeyboardInterrupt, ValueError):
        pass

    return default_start, default_goal
