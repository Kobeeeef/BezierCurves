import os
import json
import time

import numpy as np
import heapq
import cv2
from scipy.special import comb
import screeninfo
import matplotlib.pyplot as plt
bottomBoundary = 91
topBoundary = 1437
leftBoundary = 421
rightBoundary = 3352
fieldHeightMeters = 8.05
fieldWidthMeters = 17.55
ROBOT_METERS = 0.762
SAFE_RADIUS_METERS = 0.05
GRID_SIZE = (rightBoundary - leftBoundary, topBoundary - bottomBoundary)

json_filename = "static_obstacles_field.json"
static_obstacles = set()
if os.path.exists(json_filename):
    with open(json_filename, "r") as f:
        try:
            loaded_pixels = json.load(f)
            static_obstacles = set(tuple(p) for p in loaded_pixels)
            print(f"Loaded {len(static_obstacles)} pixels from {json_filename}")
        except json.JSONDecodeError:
            print("Error loading JSON file. Starting fresh.")

# ---------------------------
# Define the PathPlanner class
# ---------------------------
class PathPlanner:
    def __init__(self, grid_size, raw_obstacles, safety_radius):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=np.uint8)
        t = time.monotonic()
        for ox, oy in raw_obstacles:
            if 0 <= ox < grid_size[0] and 0 <= oy < grid_size[1]:
                self.grid[ox, oy] = 1
        print(f"Finished in {time.monotonic() - t:.2f} seconds.")
        self.obstacles = self.inflate_obstacles(self.grid, safety_radius)
        print(f"Finished in {time.monotonic() - t:.2f} seconds.")

    def inflate_obstacles(self, grid, radius):
        """Uses OpenCV to inflate obstacles with a circular kernel."""
        kernel_size = int(radius) + 2  # small safety offset
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        inflated_grid = cv2.dilate(grid.astype(np.uint8), kernel, iterations=1)
        inflated_obstacles = set(zip(*np.where(inflated_grid == 1)))
        return inflated_obstacles

    def heuristic(self, a, b):
        """Using Manhattan (diagonal) distance as a heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

        # D = 1
        # D2 = 1.414
        # dx = abs(a[0] - b[0])
        # dy = abs(a[1] - b[1])
        # return D * (dx + dy) + (D2 - 2 * D) * min(dx, dy)

    def a_star(self, start, goal):
        """A* Pathfinding Algorithm."""
        neighbors = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-way movement
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
                        neighbor not in self.obstacles):
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

    def generate_safe_bezier_paths(self, control_points):
        """
        Build segments of Bézier curves from control_points. Instead of splitting immediately when
        a collision is detected, try to inflate the segment (i.e. create a larger curve) that avoids
        the obstacle. If inflation fails, then split the segment as before.
        """
        segments = []
        segment = [control_points[0]]

        for i in range(1, len(control_points)):
            segment.append(control_points[i])
            # Compute the curve for the current segment
            curve = self.bezier_curve(segment, num_points=100)
            if self.check_collision(curve):
                # Attempt to inflate the current segment
                inflated_segment = self.try_inflate_segment(segment, max_offset=200)
                if inflated_segment is not None:
                    # Replace the current segment with the inflated version
                    # (you might decide to store both versions or mark them as adjusted)
                    segment = inflated_segment
                    # Optionally, recompute the curve and verify collision again:
                    curve = self.bezier_curve(segment, num_points=100)
                    if self.check_collision(curve):
                        # Even the inflated version still collides, so we split.
                        segments.append(segment[:-1])
                        segment = [control_points[i - 1], control_points[i]]
                else:
                    # Inflation failed; split the segment as a fallback.
                    segments.append(segment[:-1])
                    segment = [control_points[i - 1], control_points[i]]

        segments.append(segment)
        # Return segments as numpy arrays (for plotting, etc.)
        return [np.array(seg) for seg in segments]

    def try_inflate_segment(self, segment, max_offset=100, step=5):
        """
        Attempt to modify (inflate) the segment by replacing the middle control point(s)
        with an offset point based on the endpoints, in order to bend the curve away from obstacles.
        Returns a new control polygon (list of points) if a safe inflation is found,
        otherwise returns None.

        For simplicity, this example uses just the first and last point of the segment to
        create a quadratic Bézier candidate.
        """
        # For inflation, we require at least two points (start and end)
        if len(segment) < 2:
            return None

        p0 = np.array(segment[0])
        p_end = np.array(segment[-1])
        chord = p_end - p0
        chord_length = np.linalg.norm(chord)
        if chord_length == 0:
            return None
        perp = np.array([-chord[1], chord[0]]) / chord_length
        for sign in [1, -1]:
            for offset in np.arange(step, max_offset + step, step):
                # Create a candidate mid control point by offsetting the chord's midpoint.
                mid = (p0 + p_end) / 2 + sign * perp * offset
                candidate_segment = [segment[0], tuple(mid), segment[-1]]
                candidate_curve = self.bezier_curve(candidate_segment, num_points=100)
                if not self.check_collision(candidate_curve):
                    # If the candidate curve avoids obstacles, return this new control polygon.
                    return candidate_segment
        return None



# ---------------------------
# Main function
# ---------------------------
def main():
    pose2dStart = (1.21, 2.16, 0)
    pose2dGoal = (16.32, 6.51, 0)
    pixelsPerMeterX = GRID_SIZE[0] / fieldWidthMeters
    pixelsPerMeterY = GRID_SIZE[1] / fieldHeightMeters
    robotSizePixels = int(ROBOT_METERS * pixelsPerMeterX)
    safeDistancePixels = int(robotSizePixels + (SAFE_RADIUS_METERS * pixelsPerMeterX))

    startPositionPixelsX = int(pose2dStart[0] * pixelsPerMeterX)
    startPositionPixelsY = int(pose2dStart[1] * pixelsPerMeterY)
    goalPositionPixelsX = int(pose2dGoal[0] * pixelsPerMeterX)
    goalPositionPixelsY = int(pose2dGoal[1] * pixelsPerMeterY)
    print(f"StartPositionPixels: {startPositionPixelsX}, {startPositionPixelsY}")
    print(f"GoalPositionPixels: {goalPositionPixelsX}, {goalPositionPixelsY}")
    print(f"SafeDistancePixels: {safeDistancePixels}")
    print(f"RobotSizePixels: {robotSizePixels}")
    planner = PathPlanner(GRID_SIZE, static_obstacles, safety_radius=safeDistancePixels)
    print("Planning now.")
    t = time.monotonic()
    a_star_path = planner.a_star((startPositionPixelsX, startPositionPixelsY),
                                 (goalPositionPixelsX, goalPositionPixelsY))
    print(f"Path planning time: {time.monotonic() - t}")
    if not a_star_path:
        print("No path found from start to goal.")
        return

    print("Path found, now solving Bézier curve...")
    inflection_points = planner.find_inflection_points(a_star_path)
    control_points = planner.insert_midpoints(inflection_points)
    safe_paths = planner.generate_safe_bezier_paths(control_points)

    # The safe paths returned are in pixel coordinates.
    # (The external conversion from pixels to meters remains as before.)
    scaled_safe_paths = [
        (segment / np.array([pixelsPerMeterX, pixelsPerMeterY])).tolist()
        for segment in safe_paths
    ]
    print(scaled_safe_paths)


if __name__ == '__main__':
    main()
