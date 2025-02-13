import matplotlib.pyplot as plt
import os
import json
import time
import numpy as np
import cv2
from scipy.special import comb

# Field dimensions in meters
fieldHeightMeters = 8.05
fieldWidthMeters  = 17.55

# ---------------------------
# Load Exported Obstacles
# ---------------------------
json_filename = "static_obstacles_wall.json"
static_obstacles = set()
if os.path.exists(json_filename):
    with open(json_filename, "r") as f:
        try:
            loaded_pixels = json.load(f)
            # Obstacles are given as pixel coordinates with (0,0) at bottom left.
            static_obstacles = set(tuple(p) for p in loaded_pixels)
            print(f"Loaded {len(static_obstacles)} pixels from {json_filename}")
        except json.JSONDecodeError:
            print("Error loading JSON file. Starting fresh.")

# Optionally, add bounding edges (make sure indices are in range)
def add_bounding_edges(static_obs, grid_size):
    width, height = grid_size
    top_edge    = {(x, height-1) for x in range(width)}
    bottom_edge = {(x, 0)         for x in range(width)}
    left_edge   = {(0, y)         for y in range(height)}
    right_edge  = {(width-1, y)   for y in range(height)}
    static_obs.update(top_edge, bottom_edge, left_edge, right_edge)

# Uncomment the following line if you want to block the boundaries:
# GRID_SIZE = (690, 316)
# add_bounding_edges(static_obstacles, GRID_SIZE)

# ---------------------------
# Define the PathPlanner Class
# ---------------------------
class PathPlanner:
    def __init__(self, grid_size, raw_obstacles, safety_radius):
        self.grid_size = grid_size
        # Use floats for pixel-per-meter conversion
        self.pixelsPerMeterX = float(grid_size[0]) / fieldWidthMeters
        self.pixelsPerMeterY = float(grid_size[1]) / fieldHeightMeters

        # Build a grid (with (0,0) at bottom left)
        self.grid = np.zeros(grid_size, dtype=np.uint8)
        t0 = time.monotonic()
        for ox, oy in raw_obstacles:
            if 0 <= ox < grid_size[0] and 0 <= oy < grid_size[1]:
                self.grid[ox, oy] = 1
        print(f"Grid built in {time.monotonic() - t0:.2f} seconds.")

        # Inflate obstacles (using a circular kernel) to account for safety radius.
        self.obstacles = self.inflate_obstacles(self.grid, safety_radius)
        print(f"Obstacle inflation completed in {time.monotonic() - t0:.2f} seconds.")

    def inflate_obstacles(self, grid, radius):
        kernel_size = int(radius) + 2  # add a small safety offset
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        inflated = cv2.dilate(grid.astype(np.uint8), kernel, iterations=1)
        inflated_obs = set(zip(*np.where(inflated == 1)))
        return inflated_obs

    # ---------------------------
    # Bézier Evaluation
    # ---------------------------
    def bezier_curve(self, control_points, num_points=300):
        """
        Evaluates the Bézier curve for a list of control points.
        Returns an array of shape (num_points,2).
        """
        cp = np.array(control_points, dtype=float)
        n = len(cp) - 1
        t_vals = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 2), dtype=float)
        for i in range(n+1):
            basis = comb(n, i) * (1-t_vals)**(n-i) * (t_vals**i)
            curve += np.outer(basis, cp[i])
        return curve

    # ---------------------------
    # Collision Checking Utilities
    # ---------------------------
    def check_collision(self, curve):
        """
        Returns True if any sample point on the curve is in an obstacle.
        """
        for (px, py) in curve:
            ix, iy = int(round(px)), int(round(py))
            if (ix, iy) in self.obstacles:
                return True
        return False

    def count_collisions(self, curve):
        """
        Returns the number of sample points on the curve that fall on obstacles.
        """
        count = 0
        for (px, py) in curve:
            ix, iy = int(round(px)), int(round(py))
            if (ix, iy) in self.obstacles:
                count += 1
        return count

    def find_first_collision_interval(self, curve):
        """
        Scans through the sampled curve and returns (i, j) as the start and end indices
        of the first contiguous block of collision points.
        Returns (None, None) if no collisions.
        """
        n = len(curve)
        i = None
        for idx, (px,py) in enumerate(curve):
            if (int(round(px)), int(round(py))) in self.obstacles:
                i = idx
                break
        if i is None:
            return None, None
        # Find the last index j where collision continues
        j = i
        while j+1 < n:
            px, py = curve[j+1]
            if (int(round(px)), int(round(py))) in self.obstacles:
                j += 1
            else:
                break
        return i, j

    # ---------------------------
    # Bend a Collision Segment
    # ---------------------------
    def find_bend_point(self, p_i, p_j, step=1.0):
        """
        Given two points p_i and p_j (as numpy arrays) representing the endpoints
        of a collision segment along the current Bézier curve, this function tries to
        find a new candidate point that “bends” the chord away from the obstacles.
        It does so by computing the chord’s unit perpendicular (normal) and then
        increasing an offset (in both positive and negative directions) until the 3‑point
        Bézier [p_i, candidate, p_j] is collision‑free.
        Returns the candidate point if found, or None if not.
        """
        p_i = np.array(p_i, dtype=float)
        p_j = np.array(p_j, dtype=float)
        chord = p_j - p_i
        L = np.linalg.norm(chord)
        if L < 1e-6:
            return None
        # Unit normal (two possible directions)
        n = np.array([-chord[1], chord[0]], dtype=float)
        n = n / np.linalg.norm(n)
        # Try both directions until one yields a collision-free sub-curve.
        # We do not set an artificial "max" offset here; we iterate until we find one.
        # (We include a very high iteration limit to prevent infinite loops.)
        for direction in [1, -1]:
            offset = step  # start with a small offset
            for k in range(1, 10000):
                candidate = 0.5*(p_i + p_j) + direction * offset * n
                sub_curve = self.bezier_curve([p_i, candidate, p_j], num_points=50)
                if not self.check_collision(sub_curve):
                    return candidate
                offset += step
        return None

    # ---------------------------
    # Build a Collision‑Free Bézier by Inserting Control Points
    # ---------------------------
    def plan_bezier_path(self, start, goal, max_iterations=1000):
        """
        Starts with a straight-line Bézier (control points [start, goal]).
        Then it samples the curve; if any sample collides with obstacles, it finds
        the first contiguous collision segment and computes a new control point that
        “bends” that chord (using a heuristic to choose the minimal offset that
        clears obstacles). It then inserts that control point into the overall control
        polygon (using an approximate mapping from sample parameter to control-point index)
        and repeats until the curve is collision-free.
        """
        # Start with just start and goal.
        CP = [np.array(start, dtype=float), np.array(goal, dtype=float)]
        num_samples = 300
        iteration = 0
        while iteration < max_iterations:
            curve = self.bezier_curve(CP, num_points=num_samples)
            if not self.check_collision(curve):
                print(f"Found collision‑free curve after {iteration} iterations.")
                return CP
            # Find first collision interval in the sampled curve.
            i, j = self.find_first_collision_interval(curve)
            if i is None:
                # No collision found (should not happen here)
                return CP
            # Define the endpoints of the collision interval.
            p_i = curve[i]
            p_j = curve[j]
            # Attempt to find a bending candidate for the chord [p_i, p_j].
            new_pt = self.find_bend_point(p_i, p_j, step=1.0)
            if new_pt is None:
                print("Failed to find a bend point for the collision segment.")
                return None
            # Compute an approximate parameter value t for the collision.
            t_coll = (i + j) / (2.0*(num_samples-1))
            # Map t_coll (in [0,1]) to a segment index in the control polygon.
            # For a CP list of length N, assume each segment is roughly 1/(N-1) long.
            N = len(CP)
            seg_idx = int(round(t_coll * (N - 1)))
            seg_idx = max(0, min(seg_idx, N - 2))
            # Insert the new control point between CP[seg_idx] and CP[seg_idx+1].
            CP.insert(seg_idx+1, new_pt)
            iteration += 1
            # Optionally, you could print debugging info:
            print(f"Iteration {iteration}: Inserted new CP between indices {seg_idx} and {seg_idx+1}; total CPs = {len(CP)}")
        print("Reached maximum iterations without finding a collision-free path.")
        return None

# ---------------------------
# Drawing Function
# ---------------------------
def draw_results(planner, control_points):
    if control_points is None:
        print("No collision-free path found; nothing to draw.")
        return
    curve = planner.bezier_curve(control_points, num_points=300)
    plt.figure(figsize=(12, 8))
    # Plot obstacles (as black dots)
    obs = np.array(list(planner.obstacles))
    if obs.size > 0:
        plt.scatter(obs[:,0], obs[:,1], c='black', s=1, label='Obstacles')
    # Plot the Bézier curve
    plt.plot(curve[:,0], curve[:,1], 'r-', linewidth=2, label='Bézier Curve')
    # Plot control points (as green circles)
    cp_arr = np.array(control_points)
    plt.scatter(cp_arr[:,0], cp_arr[:,1], c='green', marker='o', s=50, label='Control Points')
    plt.title("Pure Bézier Path (Unlimited Control Points)")
    plt.xlabel("X (pixels)")
    plt.ylabel("Y (pixels)")
    plt.xlim(0, planner.grid_size[0])
    plt.ylim(0, planner.grid_size[1])
    plt.legend()
    plt.show()

# ---------------------------
# Main Function
# ---------------------------
def main():
    # Field/robot configuration:
    # (For example, a robot diameter of ~0.762 m; safety radius can be 0 if obstacles are already inflated.)
    ROBOT_METERS = 0.762
    SAFE_RADIUS_INCHES = 0
    SAFE_RADIUS_METERS = SAFE_RADIUS_INCHES * 0.0254

    GRID_SIZE = (690, 316)
    # Compute pixels per meter (using X dimension)
    px_per_m = float(GRID_SIZE[0]) / fieldWidthMeters

    # Start & goal in meters:
    pose2dStart = (2, 4)
    pose2dGoal  = (16, 4)
    start_px = np.array([pose2dStart[0] * px_per_m, pose2dStart[1] * px_per_m])
    goal_px  = np.array([pose2dGoal[0]  * px_per_m, pose2dGoal[1]  * px_per_m])

    # Create the planner (passing a safety radius in pixels)
    planner = PathPlanner(GRID_SIZE, static_obstacles, safety_radius=SAFE_RADIUS_METERS * px_per_m)

    t0 = time.monotonic()
    CP = planner.plan_bezier_path(start_px, goal_px, max_iterations=1000)
    elapsed = time.monotonic() - t0
    if CP is None:
        print("No collision‑free path found.")
    else:
        print(f"Collision‑free path found in {elapsed:.2f} seconds with {len(CP)} control points.")
    draw_results(planner, CP)

if __name__ == "__main__":
    main()
