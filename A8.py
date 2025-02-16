import time

import cv2
import numpy as np
import skfmm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import json


class FastMarchingPathfinder:
    def __init__(self, grid_cost):
        """
        grid_cost: 2D numpy array of base traversal costs.
                   Free cells: cost 1; obstacles: higher cost (e.g., 30, 100, 1000).
                   All values must be > 0.
        """
        self.grid_cost = grid_cost.copy()
        self.height, self.width = grid_cost.shape

    def compute_time_map(self, goal):
        """
        Compute the travel time (cost-to-go) from every cell to the goal using the Fast Marching Method.
        The speed function is defined as the reciprocal of grid_cost.
        """
        speed = 1.0 / self.grid_cost  # higher cost => lower speed
        phi = np.ones_like(self.grid_cost)
        goal_x, goal_y = goal
        phi[goal_y, goal_x] = -1
        time_map = skfmm.travel_time(phi, speed)
        return time_map

    def next_step(self, pos, time_map):
        """
        Given current position pos, return the neighbor (8-neighbors) with the lowest travel time.
        """
        x, y = pos
        best = pos
        best_time = time_map[y, x]
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1),
                       (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height:
                if time_map[ny, nx] < best_time:
                    best_time = time_map[ny, nx]
                    best = (nx, ny)
        return best

    def bezier_curve(self, control_points, num_points=100):
        """
        Generate a Bézier curve from a list of control points.
        For 2, 3, or 4 points, use the standard linear/quadratic/cubic formulas.
        For more than 4 control points, use the de Casteljau algorithm.
        """
        # Ensure all control points are NumPy arrays (floats)
        control_points = [np.array(pt, dtype=float) for pt in control_points]
        n = len(control_points)
        t_values = np.linspace(0, 1, num_points)

        if n == 2:
            # Linear Bézier (2 control points)
            curve = np.outer(1 - t_values, control_points[0]) + np.outer(t_values, control_points[1])
        elif n == 3:
            # Quadratic Bézier (3 control points)
            p0, p1, p2 = control_points
            curve = (np.outer((1 - t_values) ** 2, p0) +
                     np.outer(2 * (1 - t_values) * t_values, p1) +
                     np.outer(t_values ** 2, p2))
        elif n == 4:
            # Cubic Bézier (4 control points)
            p0, p1, p2, p3 = control_points
            curve = (np.outer((1 - t_values) ** 3, p0) +
                     np.outer(3 * t_values * (1 - t_values) ** 2, p1) +
                     np.outer(3 * t_values ** 2 * (1 - t_values), p2) +
                     np.outer(t_values ** 3, p3))
        else:
            # For any number > 4, use the de Casteljau algorithm.
            curve_points = []
            for tt in t_values:
                pts = control_points.copy()
                # Iteratively blend the points.
                for r in range(1, n):
                    pts = [(1 - tt) * pts[i] + tt * pts[i + 1] for i in range(len(pts) - 1)]
                curve_points.append(pts[0])
            curve = np.array(curve_points)

        return curve

    def check_collision(self, curve):
        """
        Check whether any point in the curve collides with an obstacle.
        We assume a collision if the grid cost at that point is above a threshold (here, 100000).
        """
        for pt in curve:
            x, y = int(round(pt[0])), int(round(pt[1]))
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue
            if self.grid_cost[y, x] >= 2:
                return True
        return False

    def try_inflate_segment(self, segment, max_offset_pixels=25, step_pixels=5):
        """
        Attempt to modify (inflate) the segment by replacing the middle control point(s)
        with an offset point (based on the endpoints) to bend the curve away from obstacles.
        Returns a new control polygon if a safe inflation is found, otherwise returns None.
        """
        if len(segment) < 2:
            return None

        p0 = np.array(segment[0])
        p_end = np.array(segment[-1])
        chord = p_end - p0
        chord_length = np.linalg.norm(chord)
        if chord_length == 0:
            return None
        # Compute a unit vector perpendicular to the chord.
        perp = np.array([-chord[1], chord[0]]) / chord_length
        for sign in [1, -1]:
            for offset in np.arange(step_pixels, max_offset_pixels + step_pixels, step_pixels):
                mid = (p0 + p_end) / 2 + sign * perp * offset
                candidate_segment = [segment[0], tuple(mid), segment[-1]]
                candidate_curve = self.bezier_curve(candidate_segment, num_points=100)
                if not self.check_collision(candidate_curve):
                    return candidate_segment
        return None

    def generate_safe_bezier_paths(self, control_points):
        """
        Build segments of Bézier curves from control_points.
        Instead of immediately splitting a segment when a collision is detected, try to inflate
        the segment to avoid the obstacle. If inflation fails, then split the segment.
        Returns:
            final_segments (list of np.ndarray): Each element is an array of control points for the segment.
        """
        segments = []
        segment = [control_points[0]]

        for i in range(1, len(control_points)):
            segment.append(control_points[i])
            curve = self.bezier_curve(segment, num_points=100)

            if self.check_collision(curve):
                # Attempt to inflate the current segment
                inflated_segment = self.try_inflate_segment(segment)
                if inflated_segment is not None:
                    segment = inflated_segment
                    curve = self.bezier_curve(segment, num_points=100)
                    if self.check_collision(curve):
                        segments.append(segment[:-1])
                        segment = [control_points[i - 1], control_points[i]]
                else:
                    segments.append(segment[:-1])
                    segment = [control_points[i - 1], control_points[i]]

        segments.append(segment)
        final_segments = [np.array(seg) for seg in segments]
        return final_segments


def get_static_obstacles(filename):
    """
    Load static obstacle coordinates from a JSON file.
    The JSON file should contain a list of [x, y] coordinates.
    Returns a list of [x, y] pairs.
    """
    with open(filename, 'r') as f:
        obstacles = json.load(f)
    return obstacles


def apply_and_inflate_all_obstacles(grid, static_obs_array, dynamic_obs_array, safe_distance):
    """
    Applies both static and dynamic obstacles (with inflation) onto the grid.
    """
    # --- Apply static obstacles ---
    for coord in static_obs_array:
        x, y = coord
        if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
            grid[y, x] = 100000

    # Inflate static obstacles using dilation.
    binary_static = (grid > 1).astype(np.uint8)
    kernel_size = int(safe_distance)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    inflated_static = cv2.dilate(binary_static, kernel, iterations=1)
    grid[inflated_static == 1] = 100000

    # --- Apply dynamic obstacles ---
    for obs in dynamic_obs_array:
        cx, cy, heat, size = obs
        mask = np.zeros_like(grid, dtype=np.uint8)
        x_min = max(0, int(np.floor(cx - size)))
        x_max = min(grid.shape[1] - 1, int(np.ceil(cx + size)))
        y_min = max(0, int(np.floor(cy - size)))
        y_max = min(grid.shape[0] - 1, int(np.ceil(cy + size)))
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= size ** 2:
                    mask[y, x] = 1
        kernel_dynamic_size = int(size) + 1
        kernel_dynamic = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_dynamic_size, kernel_dynamic_size))
        inflated_mask = cv2.dilate(mask, kernel_dynamic, iterations=1)
        grid[inflated_mask == 1] = heat

    return grid


def find_inflection_points(path):
    """
    Extract inflection points from the path (points where the direction changes).
    """
    if len(path) < 3:
        return path
    inflection_points = [path[0]]
    for i in range(1, len(path) - 1):
        prev_dx = path[i][0] - path[i - 1][0]
        prev_dy = path[i][1] - path[i - 1][1]
        next_dx = path[i + 1][0] - path[i][0]
        next_dy = path[i + 1][1] - path[i][1]
        if (prev_dx, prev_dy) != (next_dx, next_dy):
            inflection_points.append(path[i])
    inflection_points.append(path[-1])
    return inflection_points


def visualize(grid_cost, time_map, path, start, goal, bezier_segments=None, inflection_points=None, pathfinder=None):
    """
    Visualize the grid along with the discrete path, inflection points, and safe Bézier curves.
    Free cells (cost == 1) appear white, obstacles are shown via a colormap.
    """
    plt.figure(figsize=(10, 8), facecolor='white')
    extent = [0, grid_cost.shape[1], 0, grid_cost.shape[0]]
    display_map = np.where(grid_cost <= 1, np.nan, time_map)

    custom_colors = [
        (1.0, 1.0, 0.8),  # light yellow
        (1.0, 0.9, 0.0),  # orange
        (1.0, 0.0, 0.0),  # red
        (0.5, 0.0, 0.5)   # purple
    ]
    custom_cmap = LinearSegmentedColormap.from_list("heat_custom", custom_colors, N=256)
    cmap = custom_cmap.copy()
    cmap.set_bad('white')

    mask = grid_cost > 1
    if np.any(mask):
        vmin = np.nanmin(display_map[mask])
        vmax = np.nanmax(display_map[mask])
    else:
        vmin, vmax = 0, 1

    plt.imshow(display_map, cmap=cmap, extent=extent, origin='lower', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Obstacle Heat')

    # Plot the discrete path
    if path:
        xs, ys = zip(*path)
        plt.plot(xs, ys, color='red', linewidth=2, label='Discrete Path')
        plt.scatter(xs[0], ys[0], color='blue', edgecolors='black', s=100, label='Start')
        plt.scatter(xs[-1], ys[-1], color='cyan', edgecolors='black', s=100, label='Goal')

    # Plot inflection points if provided
    if inflection_points is not None:
        ip = np.array(inflection_points)
        plt.plot(ip[:, 0], ip[:, 1], 'ro--', label='Inflection Points')

    # Plot safe Bézier curves if provided
    if bezier_segments is not None and pathfinder is not None:
        for i, seg in enumerate(bezier_segments):
            curve = pathfinder.bezier_curve(seg, num_points=100)
            label = 'Safe Bézier Curve' if i == 0 else None
            plt.plot(curve[:, 0], curve[:, 1], 'b-', linewidth=2, label=label)

    plt.title("Obstacle Heatmap with Paths")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(0, grid_cost.shape[1])
    plt.ylim(0, grid_cost.shape[0])
    plt.legend()
    plt.show()


# --- Main execution (everything is in centimeters) ---
if __name__ == '__main__':
    grid_width = 1755
    grid_height = 805
    ROBOT_SIZE_CM = 89
    SAFE_DISTANCE_CM = 5
    TOTAL_SAFE_DISTANCE = ROBOT_SIZE_CM + SAFE_DISTANCE_CM

    # Create the base grid (free cells = 1)
    base_grid = np.ones((grid_height, grid_width), dtype=float)

    # Load static obstacles from file
    static_obs_array = get_static_obstacles("static_obstacles_cm.json")
    # Define dynamic obstacles: each tuple is (X, Y, HEAT, SIZE)
    dynamic_obs_array = [(800, 150, 40000, 1)]

    # Apply both static and dynamic obstacles (with inflation)
    combined_grid = apply_and_inflate_all_obstacles(base_grid.copy(), static_obs_array, dynamic_obs_array, TOTAL_SAFE_DISTANCE)

    pathfinder = FastMarchingPathfinder(combined_grid)
    start = (0, 400)
    goal = (250, 400)
    print("Computing pathfinder...")
    t = time.time()
    time_map = pathfinder.compute_time_map(goal)
    print("Time to compute pathfinder:", time.time() - t)
    print("Computed time_map")

    # Generate the discrete path using gradient descent on the time_map
    path = [start]
    current = start
    max_steps = 10000
    for _ in range(max_steps):
        next_cell = pathfinder.next_step(current, time_map)
        if next_cell == current:
            break  # No progress: local minimum reached.
        path.append(next_cell)
        current = next_cell
        if current == goal:
            break

    print(f"Path from {start} to {goal} has {len(path)} steps.")
    t = time.time()
    # Extract inflection points from the discrete path
    inflection_points = find_inflection_points(path)
    print("Time to find inflection points:", time.time() - t)
    print("Inflection points count:", len(inflection_points))
    print("Point deflation percentage:", ((len(path) - len(inflection_points)) / len(path))*100)

    t = time.time()
    # Generate safe, smooth Bézier segments from the inflection points
    safe_bezier_segments = pathfinder.generate_safe_bezier_paths(inflection_points)
    print("Time to generate safe bezier paths:", time.time() - t)

    # Visualize everything together: obstacle heatmap, discrete path, inflection points, and safe Bézier curves.
    visualize(combined_grid, time_map, path, start, goal,
              bezier_segments=safe_bezier_segments,
              inflection_points=inflection_points,
              pathfinder=pathfinder)
