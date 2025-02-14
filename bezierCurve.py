import matplotlib.pyplot as plt
import os
import json
import time
import numpy as np
import heapq
import cv2
from scipy.special import comb
import zmq

import BezierCurve_pb2 as BezierCurve

# Field dimensions in meters
fieldHeightMeters = 8.05
fieldWidthMeters = 17.55

# ---------------------------
# Load Exported Obstacles
# ---------------------------
json_filename = "static_obstacles.json"
static_obstacles = set()
if os.path.exists(json_filename):
    with open(json_filename, "r") as f:
        try:
            loaded_pixels = json.load(f)
            # These obstacles should have been exported with Y flipped so that (0,0) is bottom left.
            static_obstacles = set(tuple(p) for p in loaded_pixels)
            print(f"Loaded {len(static_obstacles)} pixels from {json_filename}")
        except json.JSONDecodeError:
            print("Error loading JSON file. Starting fresh.")


# ---------------------------
# Define the PathPlanner Class
# ---------------------------
class PathPlanner:
    def __init__(self, grid_size, raw_obstacles, safety_radius):
        self.grid_size = grid_size
        # Compute the number of pixels per meter in X and Y for the field-relative grid.
        self.pixelsPerMeterX = grid_size[0] / fieldWidthMeters
        self.pixelsPerMeterY = grid_size[1] / fieldHeightMeters
        # Create a grid (using the same coordinate system as the exported obstacles).
        self.grid = np.zeros(grid_size, dtype=np.uint8)
        self.dynamic_obstacles = []
        self.safety_radius = safety_radius
        t = time.monotonic()
        # raw_obstacles are assumed to be in field-relative coordinates where (0, 0) is bottom left.
        for ox, oy in raw_obstacles:
            if 0 <= ox < grid_size[0] and 0 <= oy < grid_size[1]:
                self.grid[ox, oy] = 1
        print(f"Grid built in {time.monotonic() - t:.2f} seconds.")
        self.obstacles = self.inflate_obstacles(self.grid, safety_radius)
        print(f"Static obstacle inflation completed in {time.monotonic() - t:.2f} seconds.")

    def setSafetyRadius(self, new_safety_radius):
        if new_safety_radius == self.safety_radius:
            return
        self.obstacles = self.inflate_obstacles(self.grid, new_safety_radius)
        self.safety_radius = new_safety_radius

    def inflate_obstacles(self, grid, radius):
        """Uses OpenCV to inflate obstacles with a circular kernel."""
        kernel_size = int(radius) + 2  # small safety offset
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        inflated_grid = cv2.dilate(grid.astype(np.uint8), kernel, iterations=1)
        inflated_obstacles = set(zip(*np.where(inflated_grid == 1)))
        return inflated_obstacles

    def heuristic(self, a, b):
        """Using Manhattan (diagonal) distance as a heuristic."""
        D = 1
        D2 = 1.414
        dx = abs(a[0] - b[0])
        dy = abs(a[1] - b[1])
        return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

    def a_star(self, start, goal):
        """A* Pathfinding Algorithm."""
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0),
                     (1, 1), (1, -1), (-1, 1), (-1, -1)]
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]

            for dx, dy in neighbors:
                neighbor = (current[0] + dx, current[1] + dy)
                if (0 <= neighbor[0] < self.grid_size[0] and
                        0 <= neighbor[1] < self.grid_size[1] and
                        neighbor not in self.obstacles and
                        neighbor not in self.dynamic_obstacles):
                    tentative_g_score = g_score[current] + 1
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return []

    def find_inflection_points(self, path):
        """Extract inflection points from the path (where direction changes)."""
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

    def insert_midpoints(self, points):
        """Insert midpoints between points for smoother curves."""
        new_points = []
        for i in range(len(points) - 1):
            new_points.append(points[i])
            midpoint = ((points[i][0] + points[i + 1][0]) / 2,
                        (points[i][1] + points[i + 1][1]) / 2)
            new_points.append(midpoint)
        new_points.append(points[-1])
        return np.array(new_points)

    def bezier_curve(self, control_points, num_points=100):
        """Compute a Bézier curve from control points."""
        n = len(control_points) - 1
        t = np.linspace(0, 1, num_points)
        curve = np.zeros((num_points, 2))
        for i in range(n + 1):
            bernstein_poly = comb(n, i) * (1 - t) ** (n - i) * t ** i
            curve += np.outer(bernstein_poly, control_points[i])
        return curve

    def check_collision(self, curve):
        """Check if any point on the Bézier curve collides with an obstacle."""
        for px, py in curve:
            if (round(px), round(py)) in self.obstacles:
                return True
        return False

    def try_inflate_segment(self, segment, max_offset_meters=2, step_meters=0.03):
        """
        Attempt to modify (inflate) the segment by replacing the middle control point(s)
        with an offset based on the endpoints, in order to bend the curve away from obstacles.
        Returns a new control polygon (list of points) if a safe inflation is found,
        otherwise returns None.
        """
        if len(segment) < 2:
            return None
        max_offset_pixels = int(max_offset_meters * self.pixelsPerMeterX)
        step_pixels = int(step_meters * self.pixelsPerMeterX)

        p0 = np.array(segment[0])
        p_end = np.array(segment[-1])
        chord = p_end - p0
        chord_length = np.linalg.norm(chord)
        if chord_length == 0:
            return None
        perp = np.array([-chord[1], chord[0]]) / chord_length
        for sign in [1, -1]:
            for offset in np.arange(step_pixels, max_offset_pixels + step_pixels, step_pixels):
                mid = (p0 + p_end) / 2 + sign * perp * offset
                candidate_segment = [segment[0], tuple(mid), segment[-1]]
                candidate_curve = self.bezier_curve(candidate_segment, num_points=100)
                if not self.check_collision(candidate_curve):
                    return candidate_segment
        return None

    def adjust_control_points(self, control_points):
        """
        For each interior control point (ignoring start and end),
        attempt to inflate the segment formed with its neighbors.
        """
        new_points = control_points.copy()
        for j in range(1, len(control_points) - 1):
            seg = [tuple(control_points[j - 1]), tuple(control_points[j]), tuple(control_points[j + 1])]
            candidate = self.try_inflate_segment(seg)
            if candidate is not None:
                # Update the middle control point to the new inflated value.
                new_points[j] = candidate[1]
        return new_points

    def generate_single_bezier_path(self, control_points, num_points=200, max_attempts=10):
        """
        Compute a single Bézier curve from all provided control points.
        If the resulting curve collides with obstacles, attempt to adjust (inflate)
        the interior control points to steer the curve away from obstacles.
        """
        # Start with the original control points (assumed to be in pixel coordinates).
        inflated_points = np.array(control_points, dtype=float)
        for attempt in range(max_attempts):
            # Insert midpoints for smoother curves.
            refined_points = self.insert_midpoints(inflated_points)
            curve = self.bezier_curve(refined_points, num_points=num_points)
            if not self.check_collision(curve):
                print(f"Collision-free curve found after {attempt} inflation attempt(s).")
                return curve, inflated_points
            print(f"Collision detected on attempt {attempt + 1}. Inflating control points...")
            # Try to adjust each interior control point.
            inflated_points = self.adjust_control_points(inflated_points)
        print("Maximum inflation attempts reached; returning the last computed curve (may still collide).")
        refined_points = self.insert_midpoints(inflated_points)
        return self.bezier_curve(refined_points, num_points=num_points), inflated_points

    def set_dynamic_obstacles(self, dynamic_obstacles, safety_radius):
        """Update the grid with dynamic obstacles and apply inflation."""
        pixelsPerMeterX = self.grid_size[0] / fieldWidthMeters
        pixelsPerMeterY = self.grid_size[1] / fieldHeightMeters

        dynamic_grid = np.zeros(self.grid_size, dtype=np.uint8)
        for pose in dynamic_obstacles:
            ox = int(pose[0] * pixelsPerMeterX)
            oy = int(pose[1] * pixelsPerMeterY)
            if 0 <= ox < self.grid_size[0] and 0 <= oy < self.grid_size[1]:
                dynamic_grid[ox, oy] = 1

        self.dynamic_obstacles = self.inflate_obstacles(dynamic_grid, safety_radius)


# ---------------------------
# Drawing Function (Single Curve)
# ---------------------------
def draw_results_single(planner, bezier_curve, control_points):
    plt.figure(figsize=(12, 8))
    plt.plot(bezier_curve[:, 0], bezier_curve[:, 1], 'r-', linewidth=2, label='Single Bézier Curve')
    plt.title("Single Bézier Curve with Inflation and Obstacles")
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
    # ---------------------------
    # Field and Grid Configuration
    # ---------------------------
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    bind = "tcp://127.0.0.1:8531"
    socket.bind(bind)
    print("Server started on " + bind)
    GRID_SIZE = (690, 316)
    ROBOT_METERS = 0.762
    pixelsPerMeterX = GRID_SIZE[0] / fieldWidthMeters
    pixelsPerMeterY = GRID_SIZE[1] / fieldHeightMeters
    robotSizePixels = int(ROBOT_METERS * pixelsPerMeterX)
    defaultSafeInches = 5
    SAFE_RADIUS_METERS = defaultSafeInches * 0.0254
    safeDistancePixels = int(robotSizePixels + (SAFE_RADIUS_METERS * pixelsPerMeterX))
    planner = PathPlanner(GRID_SIZE, static_obstacles, safeDistancePixels)

    while True:
        message = socket.recv()
        request = BezierCurve.PlanBezierPathRequest.FromString(message)
        pose2dStart = (request.start.x, request.start.y)
        pose2dGoal = (request.goal.x, request.goal.y)
        print(pose2dStart)
        print(pose2dGoal)
        safeInches = request.safeRadiusInches
        speedMetersPerSecond = request.metersPerSecond

        # Update safety parameters
        SAFE_RADIUS_METERS = safeInches * 0.0254
        safeDistancePixels = int(robotSizePixels + (SAFE_RADIUS_METERS * pixelsPerMeterX))
        planner.setSafetyRadius(safeDistancePixels)

        # Convert start and goal from meters to pixels.
        startPositionPixelsX = int(pose2dStart[0] * pixelsPerMeterX)
        startPositionPixelsY = int(pose2dStart[1] * pixelsPerMeterY)
        goalPositionPixelsX = int(pose2dGoal[0] * pixelsPerMeterX)
        goalPositionPixelsY = int(pose2dGoal[1] * pixelsPerMeterY)

        t = time.monotonic()
        a_star_path = planner.a_star((startPositionPixelsX, startPositionPixelsY),
                                     (goalPositionPixelsX, goalPositionPixelsY))
        print(f"Path planning time: {time.monotonic() - t:.2f} seconds.")

        if not a_star_path:
            print("No path found from start to goal.")
            final_control_points_meters = None
            single_curve = None
        else:
            print("Path found, now generating a single Bézier curve with inflation if needed...")
            inflection_points = planner.find_inflection_points(a_star_path)
            control_points = np.array(inflection_points)
            single_curve, final_control_points = planner.generate_single_bezier_path(control_points, num_points=200)
            conversion_factors = np.array([planner.pixelsPerMeterX, planner.pixelsPerMeterY])
            final_control_points_meters = final_control_points / conversion_factors

        bezier_curves_msg = BezierCurve.BezierCurve()
        if final_control_points_meters is not None and single_curve is not None:
            for (x_val, y_val) in final_control_points_meters:
                cp = bezier_curves_msg.controlPoints.add()
                cp.x = x_val
                cp.y = y_val

            # Convert the full Bézier curve from pixels to meters.
            single_curve_meters = single_curve / conversion_factors
            # Compute the curve length as the sum of Euclidean distances between consecutive points.
            distances = np.linalg.norm(np.diff(single_curve_meters, axis=0), axis=1)
            total_length = np.sum(distances)
            # Calculate time to traverse given the speed (meters per second).
            timeToTraverse = total_length / speedMetersPerSecond
            bezier_curves_msg.timeToTraverse = timeToTraverse
            print(f"Total curve length (meters): {total_length:.2f}")
            print(f"Time to traverse (seconds): {timeToTraverse:.2f}")
        else:
            # No valid path found.
            bezier_curves_msg.timeToTraverse = -1.0

        socket.send(bezier_curves_msg.SerializeToString(), zmq.DONTWAIT)


if __name__ == '__main__':
    main()
