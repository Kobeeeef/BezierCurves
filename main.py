import os
import json
import numpy as np
import heapq
import cv2
from scipy.special import comb
from enum import Enum

# --- Global definitions (all in pixels for the ROI) ---
bottomBoundary = 91
topBoundary = 1437
leftBoundary = 421
rightBoundary = 3352

fieldHeightMeters = 8.05
fieldWidthMeters = 17.55
ROBOT_METERS = 0.762
SAFE_RADIUS_METERS = 0.05

# GRID_SIZE is defined as the pixel dimensions of the region of interest.
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
    class Accuracy(Enum):
        PIXELS = 1
        INCHES = 2

    def __init__(self, grid_size, raw_obstacles, safety_radius, accuracy=Accuracy.PIXELS):
        """
        grid_size: tuple in pixels (width, height)
        raw_obstacles: set of (x, y) in pixel coordinates
        safety_radius: in pixels
        accuracy: either Accuracy.PIXELS or Accuracy.INCHES.
                  When set to INCHES, the planner converts all internal computations to inches.
        """
        self.accuracy = accuracy

        # Compute pixels per meter for the ROI (assumed to cover the full field)
        pixelsPerMeterX = grid_size[0] / fieldWidthMeters
        pixelsPerMeterY = grid_size[1] / fieldHeightMeters

        if self.accuracy == PathPlanner.Accuracy.INCHES:
            # Conversion factor: 1 meter ≈ 39.37 inches.
            inches_per_meter = 39.37

            # Compute pixels per inch in each dimension.
            pixelsPerInchX = pixelsPerMeterX / inches_per_meter
            pixelsPerInchY = pixelsPerMeterY / inches_per_meter
            self._pixelsPerInch = np.array([pixelsPerInchX, pixelsPerInchY])

            # Convert the grid size from pixels to inches.
            grid_size = (int(round(grid_size[0] / pixelsPerInchX)),
                         int(round(grid_size[1] / pixelsPerInchY)))

            # Convert raw obstacles from pixel coordinates to inches.
            new_raw_obstacles = set()
            for ox, oy in raw_obstacles:
                new_ox = int(round(ox / pixelsPerInchX))
                new_oy = int(round(oy / pixelsPerInchY))
                new_raw_obstacles.add((new_ox, new_oy))
            raw_obstacles = new_raw_obstacles

            # Convert the safety radius from pixels to inches.
            safety_radius = safety_radius / pixelsPerInchX  # use X dimension (assumed similar)

        else:
            # When planning in pixels, no conversion is necessary.
            self._pixelsPerInch = np.array([1, 1])

        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=np.uint8)

        # Populate the grid with obstacles.
        for ox, oy in raw_obstacles:
            if 0 <= ox < grid_size[0] and 0 <= oy < grid_size[1]:
                self.grid[ox, oy] = 1

        self.obstacles = self.inflate_obstacles(self.grid, safety_radius)

    def inflate_obstacles(self, grid, radius):
        """Uses OpenCV to inflate obstacles with a circular kernel."""
        kernel_size = int(radius) + 2  # ensure integer kernel size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        inflated_grid = cv2.dilate(grid.astype(np.uint8), kernel, iterations=1)
        inflated_obstacles = set(zip(*np.where(inflated_grid == 1)))
        return inflated_obstacles

    def heuristic(self, a, b):
        """Manhattan distance heuristic."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def a_star(self, start, goal):
        """
        A* pathfinding.
        Input:
            start, goal: given in pixel coordinates.
        Internally, if planning in inches, these points are converted into inches before planning.
        """
        if self.accuracy == PathPlanner.Accuracy.INCHES:
            # Convert start and goal from pixels to inches.
            start = (int(round(start[0] / self._pixelsPerInch[0])),
                     int(round(start[1] / self._pixelsPerInch[1])))
            goal = (int(round(goal[0] / self._pixelsPerInch[0])),
                    int(round(goal[1] / self._pixelsPerInch[1])))

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
        Generate safe Bézier curve segments.
        All internal computations are done in the planning unit (pixels or inches).
        If planning in inches, the final result is converted back to pixel coordinates.
        """
        segments = []
        segment = [control_points[0]]
        for i in range(1, len(control_points)):
            segment.append(control_points[i])
            curve = self.bezier_curve(segment, num_points=100)
            if self.check_collision(curve):
                inflated_segment = self.try_inflate_segment(segment, max_offset=200)
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
        result = [np.array(seg) for seg in segments]
        if self.accuracy == PathPlanner.Accuracy.INCHES:
            # Convert each result segment from inches back to pixels.
            factor = self._pixelsPerInch  # elementwise factor (for x and y)
            result = [seg * factor for seg in result]
        return result

    def try_inflate_segment(self, segment, max_offset=100, step=5):
        """
        Try to inflate the segment by inserting an offset control point so that the curve avoids obstacles.
        Returns a new control polygon if a safe inflation is found; otherwise, returns None.
        """
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
                mid = (p0 + p_end) / 2 + sign * perp * offset
                candidate_segment = [segment[0], tuple(mid), segment[-1]]
                candidate_curve = self.bezier_curve(candidate_segment, num_points=100)
                if not self.check_collision(candidate_curve):
                    return candidate_segment
        return None

# ---------------------------
# Main function (external values remain unchanged)
# ---------------------------
def main():
    # Field start/goal in meters.
    pose2dStart = (6.2, 4, 0)
    pose2dGoal = (10, 8, 0)
    pixelsPerMeterX = GRID_SIZE[0] / fieldWidthMeters
    pixelsPerMeterY = GRID_SIZE[1] / fieldHeightMeters
    robotSizePixels = ROBOT_METERS * pixelsPerMeterX
    safeDistancePixels = robotSizePixels + (SAFE_RADIUS_METERS * pixelsPerMeterX)

    # No external conversion: start and goal are computed in pixel coordinates.
    startPositionPixelsX = pose2dStart[0] * pixelsPerMeterX
    startPositionPixelsY = pose2dStart[1] * pixelsPerMeterY
    goalPositionPixelsX = pose2dGoal[0] * pixelsPerMeterX
    goalPositionPixelsY = pose2dGoal[1] * pixelsPerMeterY
    print(f"StartPositionPixels: {startPositionPixelsX}, {startPositionPixelsY}")
    print(f"GoalPositionPixels: {goalPositionPixelsX}, {goalPositionPixelsY}")
    # Create the planner with the desired accuracy (in this example, INCHES)
    planner = PathPlanner(GRID_SIZE, static_obstacles, safety_radius=safeDistancePixels,
                          accuracy=PathPlanner.Accuracy.INCHES)

    # Call a_star with pixel coordinates. The planner handles conversion internally.
    a_star_path = planner.a_star((startPositionPixelsX, startPositionPixelsY),
                                 (goalPositionPixelsX, goalPositionPixelsY))
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
