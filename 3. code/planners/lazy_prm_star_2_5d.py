import numpy as np
import heapq

class LazyPRMStar25DNode:
    def __init__(self, idx, x, y, yaw=0.0):
        self.idx = idx
        self.x = x
        self.y = y
        self.yaw = yaw
        self.neighbors = []  # list of neighbor node indices

class LazyPRMStar25D:
    """
    2.5D LazyPRM* / ArtPlanner (Wellhausen & Hutter, 2023 - ANYbotics / ETH Zurich)
    Sampling-based roadmap planner designed for 2.5D elevation maps.
    Constructs a roadmap on the 2.5D manifold with lazy edge validation and footprint slope costing.
    """
    def __init__(self, hybrid_map, num_samples=2500, k_neighbors=18,
                 connection_radius=7.5, roll_max=0.60, pitch_max=0.60,
                 w_roll=1.0, w_pitch=0.8, w_slope=2.0):
        self.map = hybrid_map
        self.num_samples = num_samples
        self.k_neighbors = k_neighbors
        self.connection_radius = connection_radius
        self.roll_max = roll_max
        self.pitch_max = pitch_max
        self.w_roll = w_roll
        self.w_pitch = w_pitch
        self.w_slope = w_slope

        self.nodes = []
        self.invalid_edges = set()  # set of (min_idx, max_idx)

    def _sample_free(self):
        """Samples state (x, y, yaw) in non-lethal 2.5D map areas."""
        margin = 0.5
        for _ in range(50):
            x = np.random.uniform(margin, self.map.width - margin)
            y = np.random.uniform(margin, self.map.height - margin)
            if not self.map.is_occupied_2d(x, y):
                yaw = np.random.uniform(-np.pi, np.pi)
                return x, y, yaw
        return None

    def _edge_cost_and_validity(self, n1, n2):
        """
        Evaluates 3D distance, footprint roll/pitch, and slope validity along edge (n1 -> n2).
        Returns (cost, is_valid).
        """
        dist_2d = np.hypot(n2.x - n1.x, n2.y - n1.y)
        num_steps = max(int(np.ceil(dist_2d / 0.4)), 2)

        total_d3d = 0.0
        total_slope_cost = 0.0

        pts_x = np.linspace(n1.x, n2.x, num_steps)
        pts_y = np.linspace(n1.y, n2.y, num_steps)
        yaw_edge = np.arctan2(n2.y - n1.y, n2.x - n1.x)

        for i in range(num_steps):
            x_i, y_i = pts_x[i], pts_y[i]
            if self.map.is_occupied_2d(x_i, y_i):
                return float('inf'), False

            roll, pitch, _ = self.map.project_footprint_3point(x_i, y_i, yaw_edge)
            if abs(roll) > self.roll_max or abs(pitch) > self.pitch_max:
                return float('inf'), False

            if i > 0:
                z_prev = self.map.get_elevation(pts_x[i - 1], pts_y[i - 1])
                z_curr = self.map.get_elevation(x_i, y_i)
                d3 = np.sqrt((pts_x[i] - pts_x[i - 1])**2 + (pts_y[i] - pts_y[i - 1])**2 + (z_curr - z_prev)**2)
                total_d3d += d3

                slope_c = (abs(roll) * self.w_roll + abs(pitch) * self.w_pitch) * self.w_slope
                total_slope_cost += slope_c * d3

        cost = total_d3d + total_slope_cost
        return cost, True

    def plan(self, start_pose, goal_pose):
        """
        Executes 2.5D LazyPRM* from start_pose to goal_pose.
        :return: path [(x, y, z, yaw, roll, pitch, tau), ...], success (bool)
        """
        sx, sy, syaw = start_pose
        gx, gy, gyaw = goal_pose

        # Clear occupancy around start and goal
        self.map.clear_around_point(sx, sy, radius=1.0)
        self.map.clear_around_point(gx, gy, radius=1.0)

        self.nodes = []
        self.invalid_edges = set()

        # Add Start (node 0) and Goal (node 1)
        start_node = LazyPRMStar25DNode(0, sx, sy, syaw)
        goal_node = LazyPRMStar25DNode(1, gx, gy, gyaw)
        self.nodes.extend([start_node, goal_node])

        # Sample points on 2.5D manifold
        for i in range(self.num_samples):
            pt = self._sample_free()
            if pt is not None:
                x, y, yaw = pt
                node = LazyPRMStar25DNode(len(self.nodes), x, y, yaw)
                self.nodes.append(node)

        # Build initial Roadmap without collision checking (Lazy)
        coords = np.array([[n.x, n.y] for n in self.nodes])
        num_total = len(self.nodes)

        for i in range(num_total):
            dists = np.hypot(coords[:, 0] - coords[i, 0], coords[:, 1] - coords[i, 1])
            near_indices = np.argsort(dists)[1:min(self.k_neighbors + 1, num_total)]
            for j in near_indices:
                if dists[j] <= self.connection_radius:
                    if j not in self.nodes[i].neighbors:
                        self.nodes[i].neighbors.append(j)
                    if i not in self.nodes[j].neighbors:
                        self.nodes[j].neighbors.append(i)

        # Lazy Search loop: Search A* on roadmap, validate edges lazily
        max_search_attempts = 300
        attempt = 0
        success = False
        found_node_path = []

        while attempt < max_search_attempts:
            attempt += 1

            # Run A* on current unpruned roadmap
            open_set = []
            heapq.heappush(open_set, (0.0, 0))  # (f_score, node_idx)
            came_from = {}
            g_score = {0: 0.0}

            path_found = False

            while open_set:
                f, current = heapq.heappop(open_set)

                if current == 1:  # Goal reached
                    path_found = True
                    break

                curr_node = self.nodes[current]
                for nbr in curr_node.neighbors:
                    edge_key = (min(current, nbr), max(current, nbr))
                    if edge_key in self.invalid_edges:
                        continue

                    nbr_node = self.nodes[nbr]
                    est_d = np.hypot(nbr_node.x - curr_node.x, nbr_node.y - curr_node.y)
                    tentative_g = g_score[current] + est_d

                    if nbr not in g_score or tentative_g < g_score[nbr]:
                        came_from[nbr] = current
                        g_score[nbr] = tentative_g
                        h = np.hypot(goal_node.x - nbr_node.x, goal_node.y - nbr_node.y)
                        heapq.heappush(open_set, (tentative_g + h, nbr))

            if not path_found:
                break

            # Reconstruct candidate path
            candidate_indices = []
            curr = 1
            while curr in came_from:
                candidate_indices.append(curr)
                curr = came_from[curr]
            candidate_indices.append(0)
            candidate_indices.reverse()

            # Lazily validate edges along candidate path
            all_edges_valid = True
            for k in range(len(candidate_indices) - 1):
                u, v = candidate_indices[k], candidate_indices[k + 1]
                edge_key = (min(u, v), max(u, v))
                if edge_key in self.invalid_edges:
                    all_edges_valid = False
                    break

                _, is_valid = self._edge_cost_and_validity(self.nodes[u], self.nodes[v])
                if not is_valid:
                    self.invalid_edges.add(edge_key)
                    all_edges_valid = False
                    break  # Edge pruned, re-run A*

            if all_edges_valid:
                found_node_path = candidate_indices
                success = True
                break

        if not success or not found_node_path:
            # Fallback to closest reachable node path
            if came_from:
                closest_idx = min(came_from.keys(), key=lambda idx: np.hypot(self.nodes[idx].x - gx, self.nodes[idx].y - gy))
                candidate_indices = []
                curr = closest_idx
                while curr in came_from:
                    candidate_indices.append(curr)
                    curr = came_from[curr]
                candidate_indices.append(0)
                candidate_indices.reverse()
                found_node_path = candidate_indices
                success = np.hypot(self.nodes[closest_idx].x - gx, self.nodes[closest_idx].y - gy) <= 3.0
            else:
                return [], False

        # Dense trajectory interpolation along valid roadmap path
        path = []
        for k in range(len(found_node_path) - 1):
            n1 = self.nodes[found_node_path[k]]
            n2 = self.nodes[found_node_path[k + 1]]
            d2 = np.hypot(n2.x - n1.x, n2.y - n1.y)
            steps = max(int(np.ceil(d2 / 0.35)), 1)
            yaw_seg = np.arctan2(n2.y - n1.y, n2.x - n1.x)

            for s in range(steps):
                alpha = s / float(steps)
                wx = (1.0 - alpha) * n1.x + alpha * n2.x
                wy = (1.0 - alpha) * n1.y + alpha * n2.y
                wz = self.map.get_elevation(wx, wy)
                roll, pitch, _ = self.map.project_footprint_3point(wx, wy, yaw_seg)
                tau = self.map.compute_dynamic_traversability(wx, wy, yaw_seg)
                path.append((wx, wy, wz, yaw_seg, roll, pitch, tau))

        # Add final goal state
        gz = self.map.get_elevation(gx, gy)
        g_roll, g_pitch, _ = self.map.project_footprint_3point(gx, gy, gyaw)
        g_tau = self.map.compute_dynamic_traversability(gx, gy, gyaw)
        path.append((gx, gy, gz, gyaw, g_roll, g_pitch, g_tau))

        return path, success
