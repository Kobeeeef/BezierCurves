import matplotlib.pyplot as plt
import zmq

import BezierCurve_pb2 as BezierCurve
import os
import json
import time

import numpy as np
import heapq
import cv2
from scipy.special import comb

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
        self.safety_radius = safety_radius
        self.dynamic_obstacles = []
        # Compute the number of pixels per meter in X and Y for the field-relative grid.
        self.pixelsPerMeterX = grid_size[0] / fieldWidthMeters
        self.pixelsPerMeterY = grid_size[1] / fieldHeightMeters
        # Create a grid (using the same coordinate system as the exported obstacles).
        self.grid = np.zeros(grid_size, dtype=np.uint8)

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
        """A* Pathfinding Algorithm.
           (This algorithm works on the grid you provide, and since your grid is built
            with (0,0) at the bottom left, (0,0) is indeed the bottom left.)"""
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1),
                     (-1, -1)]  # 4-way movement (up, right, down, left, and corners)
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
        """Insert midpoints between inflection points for smoother curves."""
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

    def generate_safe_bezier_paths(self, control_points, speedMetersPerSecond):
        """
        Build segments of Bézier curves from control_points. Instead of splitting immediately when
        a collision is detected, try to inflate the segment (i.e. create a larger curve) that avoids
        the obstacle. If inflation fails, then split the segment as before.

        Returns:
            final_segments (list of np.ndarray): Each element is a NumPy array of points (control points).
            times_to_traverse (list of float): Each element corresponds to the time to traverse
                                               the corresponding segment at the given speed.
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

        # Append the final segment
        segments.append(segment)

        # Convert segments to NumPy arrays (for convenience/plotting)
        final_segments = [np.array(seg) for seg in segments]

        # Compute traversal time for each segment
        times_to_traverse = []
        for seg in segments:
            # Sample the final Bézier curve for this segment
            curve_points = self.bezier_curve(seg, num_points=100)

            # Calculate Euclidean distance by summing lengths between consecutive points
            total_distance = 0.0
            for j in range(1, len(curve_points)):
                total_distance += np.linalg.norm(curve_points[j] - curve_points[j - 1])

            # Time = distance / speed
            times_to_traverse.append(total_distance / speedMetersPerSecond)

        return final_segments, times_to_traverse

    def try_inflate_segment(self, segment, max_offset_meters=0.5, step_meters=0.1):
        """
        Attempt to modify (inflate) the segment by replacing the middle control point(s)
        with an offset point based on the endpoints, in order to bend the curve away from obstacles.
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


def build_bezier_curves_proto(final_segments, times_to_traverse):
    """
    Build a BezierCurves protobuf from the final path segments and their traversal times.

    Args:
        final_segments (list of np.ndarray): Each element is a NumPy array of shape (N, 2),
                                             containing [x, y] points for a segment.
        times_to_traverse (list of float): Each element is the time (in seconds) required
                                           to traverse the corresponding segment.

    Returns:
        your_proto_pb2.BezierCurves: The populated protobuf message.
    """

    # Create the top-level BezierCurves message
    bezier_curves_msg = BezierCurve.BezierCurves()
    bezier_curves_msg.pathFound = True
    # Iterate through segments and their times
    for segment, time_traverse in zip(final_segments, times_to_traverse):
        # Create a BezierCurve entry
        curve_msg = bezier_curves_msg.curves.add()
        curve_msg.timeToTraverse = time_traverse

        # Fill in the control points for this curve
        for (x_val, y_val) in segment:
            cp = curve_msg.controlPoints.add()
            cp.x = x_val
            cp.y = y_val

    return bezier_curves_msg


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
        # path planning
        SAFE_RADIUS_METERS = safeInches * 0.0254
        safeDistancePixels = int(robotSizePixels + (SAFE_RADIUS_METERS * pixelsPerMeterX))
        planner.setSafetyRadius(safeDistancePixels)
        startPositionPixelsX = int(pose2dStart[0] * pixelsPerMeterX)
        startPositionPixelsY = int(pose2dStart[1] * pixelsPerMeterY)
        goalPositionPixelsX = int(pose2dGoal[0] * pixelsPerMeterX)
        goalPositionPixelsY = int(pose2dGoal[1] * pixelsPerMeterY)
        t = time.monotonic()
        a_star_path = planner.a_star((startPositionPixelsX, startPositionPixelsY),
                                     (goalPositionPixelsX, goalPositionPixelsY))
        print(f"Path planning time: {time.monotonic() - t}")
        if not a_star_path:
            print("No path found from start to goal.")
            curves, time_to_traverse = None, None
        else:
            print("Path found, now solving Bézier curve...")
            inflection_points = planner.find_inflection_points(a_star_path)
            # control_points = planner.insert_midpoints(inflection_points)
            safe_paths, time_to_traverse = planner.generate_safe_bezier_paths(inflection_points, speedMetersPerSecond)

            curves = [
                (segment / np.array([pixelsPerMeterX, pixelsPerMeterY])).tolist()
                for segment in safe_paths
            ]
        # path planning

        if curves is None or time_to_traverse is None:
            bezier_curves_msg = BezierCurve.BezierCurves()
            bezier_curves_msg.pathFound = False
            socket.send(bezier_curves_msg.SerializeToString(), zmq.DONTWAIT)
            continue
        response = build_bezier_curves_proto(curves, time_to_traverse)
        print(response)
        socket.send(response.SerializeToString(), zmq.DONTWAIT)


if __name__ == '__main__':
    main()
