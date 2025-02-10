import os
import json
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

GRID_SIZE = (rightBoundary - leftBoundary, topBoundary - bottomBoundary)


# ---------------------------
# Define the PathPlanner class
# ---------------------------
class PathPlanner:
    def __init__(self, grid_size, raw_obstacles, safety_radius):
        self.grid_size = grid_size
        self.grid = np.zeros(grid_size, dtype=np.uint8)

        for ox, oy in raw_obstacles:
            if 0 <= ox < grid_size[0] and 0 <= oy < grid_size[1]:
                self.grid[ox, oy] = 1

        self.obstacles = self.inflate_obstacles(self.grid, safety_radius)

    def inflate_obstacles(self, grid, radius):
        """Uses OpenCV to inflate obstacles with a circular kernel."""
        kernel_size = radius + 2  # small safety offset
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
# Global variable to store click coordinates
# ---------------------------
clicks = []


def mouse_callback(event, x, y, flags, param):
    global clicks
    img = param['image']
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print("Clicked at:", (x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Field", img)


# ---------------------------
# Main function
# ---------------------------
def main():
    global clicks
    field_img = cv2.imread("2025field.png")
    if field_img is None:
        print("Error: Could not load image '2025field.png'")
        return

    img_h, img_w = field_img.shape[:2]
    if (img_w, img_h) != GRID_SIZE:
        print("Resizing image to match grid size.")
        field_img = cv2.resize(field_img, GRID_SIZE)

    img_display = field_img.copy()
    cv2.namedWindow("Field", cv2.WINDOW_FREERATIO)
    cv2.setMouseCallback("Field", mouse_callback, param={'image': img_display})

    print("Click on the image to set the START and then the GOAL point.")
    while True:
        cv2.imshow("Field", img_display)
        key = cv2.waitKey(20) & 0xFF
        if key == 27 or len(clicks) >= 2:
            break
    cv2.destroyWindow("Field")

    if len(clicks) < 2:
        print("Need two clicks (start and goal) to run path planning.")
        return
    clicks = [(725, 994), (2139, 318)]
    start = (int(clicks[0][0]), int(clicks[0][1]))
    goal = (int(clicks[1][0]), int(clicks[1][1]))
    print("Start:", start, "Goal:", goal)

    json_filename = "filled_pixels.json"
    filled_pixels = set()

    if os.path.exists(json_filename):
        with open(json_filename, "r") as f:
            try:
                loaded_pixels = json.load(f)
                filled_pixels = set(tuple(p) for p in loaded_pixels)
                print(f"Loaded {len(filled_pixels)} pixels from {json_filename}")
            except json.JSONDecodeError:
                print("Error loading JSON file. Starting fresh.")

    img_original = cv2.imread("2025field.png")
    if img_original is None:
        print("Error: Could not load image '2025field.png' for scaling obstacles.")
        return
    h_orig, w_orig = img_original.shape[:2]


    screen = screeninfo.get_monitors()[0]
    screen_w, screen_h = screen.width, screen.height

    scale_w = screen_w / w_orig
    scale_h = screen_h / h_orig
    scale_extraction = min(scale_w, scale_h, 1.0)
    new_w = int(w_orig * scale_extraction)
    new_h = int(h_orig * scale_extraction)
    print(f"Extraction image size: {new_w} x {new_h}")

    grid_w, grid_h = GRID_SIZE
    scale_x = grid_w / new_w
    scale_y = grid_h / new_h
    print(f"Scaling factors: scale_x = {scale_x}, scale_y = {scale_y}")
    scaled_obstacles = {(int(x * scale_x), int(y * scale_y)) for (x, y) in filled_pixels}

    RAW_OBSTACLES = scaled_obstacles
    planner = PathPlanner(GRID_SIZE, RAW_OBSTACLES, safety_radius=200)

    a_star_path = planner.a_star(start, goal)
    if not a_star_path:
        print("No path found from start to goal.")
        return

    inflection_points = planner.find_inflection_points(a_star_path)
    control_points = planner.insert_midpoints(inflection_points)
    safe_paths = planner.generate_safe_bezier_paths(control_points)



    fig, ax = plt.subplots(figsize=(10, 8))

    field_img_rgb = cv2.cvtColor(field_img, cv2.COLOR_BGR2RGB)

    ax.imshow(field_img_rgb, extent=[0, GRID_SIZE[0], GRID_SIZE[1], 0], origin='upper')
    a_star_path_np = np.array(a_star_path)
    ax.plot(a_star_path_np[:, 0], a_star_path_np[:, 1],
            'c-', label="A* Path", linewidth=2)
    for idx, segment in enumerate(safe_paths):

        segment = np.array(segment)

        curve = planner.bezier_curve(segment, num_points=100)

        # ax.plot(curve[:, 0], curve[:, 1],
        #         label=f"Segment {idx} Curve",
        #         linewidth=2)


        # ax.plot(segment[:, 0], segment[:, 1],
        #         'r--',
        #         label=f"Segment {idx} Control Polygon")

    ax.set_title("Safe Bézier Curve Segments on Field Map")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.legend(loc='best')
    legend = ax.get_legend()
    # if legend:
    #     legend.remove()
    plt.show()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
