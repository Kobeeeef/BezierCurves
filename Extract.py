import cv2
import numpy as np
import json
import os

#############################################
# Flood Fill at a true 1000x500 resolution.
# The image is physically resized to 1000x500.
# Bottom-left coordinate (0,0).
#############################################

json_filename = "static_obstacles_inch.json"  # Where to save your fill coordinates
image_path = "2025game-field.jpg"     # Change to your path

# Load original image
img_original = cv2.imread(image_path)
if img_original is None:
    raise ValueError("Could not load image! Check file path.")

# -- Resize to exactly 690 x 316 ---
target_w, target_h = 690, 316
img_resized = cv2.resize(
    img_original,
    (target_w, target_h),
    interpolation=cv2.INTER_AREA
)

# Now "img_resized" is our working image.
h, w = img_resized.shape[:2]

# We'll use this for on-screen display AND for actual data
img = img_resized.copy()

# Create a named window
cv2.namedWindow("Flood Fill (1000x500)", cv2.WINDOW_AUTOSIZE)

# -------------------------------------------
# Global Variables
# -------------------------------------------
filled_pixels = set()  # set of (x, y) in bottom-left coords
history = []           # stack of sets for undo

# Load existing fill data from JSON (if any)
if os.path.exists(json_filename):
    try:
        with open(json_filename, "r") as f:
            loaded = json.load(f)
            filled_pixels = set(tuple(p) for p in loaded)
            print(f"Loaded {len(filled_pixels)} pixels from {json_filename}")
    except (json.JSONDecodeError, ValueError):
        print("JSON file invalid or empty. Starting fresh.")

# Helper: Redraw existing fills
def redraw_filled_pixels():
    global img
    # Reset to the resized original
    img = img_resized.copy()
    for (lx, ly) in filled_pixels:
        # Convert bottom-left coords -> top-left coords
        col = lx
        row = (h - 1) - ly
        if 0 <= col < w and 0 <= row < h:
            img[row, col] = (0, 255, 0)

redraw_filled_pixels()
cv2.imshow("Flood Fill (1000x500)", img)

# -------------------------------------------
# Flood Fill Operation (Bucket Fill Mode)
# -------------------------------------------
def flood_fill(start_x, start_y):
    global filled_pixels, history, img
    # Copy for flood fill
    img_copy = img.copy()
    # For floodFill, we need a mask with a 1-pixel border
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    loDiff = (5, 5, 5)
    upDiff = (40, 40, 40)
    fill_color = (0, 255, 0)

    # Perform the flood fill
    cv2.floodFill(
        img_copy,
        mask,
        (start_x, start_y),
        fill_color,
        loDiff,
        upDiff,
        flags=cv2.FLOODFILL_FIXED_RANGE
    )

    # Remove the 1-pixel border from the mask
    mask = mask[1:-1, 1:-1]

    # Find filled areas
    filled_locs = np.column_stack(np.where(mask > 0))  # (rows, cols)
    new_pixels = set()
    for row, col in filled_locs:
        # Convert from top-left coords to bottom-left coords
        local_x = col
        local_y = (h - 1) - row
        new_pixels.add((local_x, local_y))

    if new_pixels:
        history.append(new_pixels)
        filled_pixels.update(new_pixels)
        img = img_copy
        cv2.imshow("Flood Fill (1000x500)", img)
        print(f"Flood filled. Total: {len(filled_pixels)}")

# -------------------------------------------
# Undo Function
# -------------------------------------------
def undo_last_fill():
    global filled_pixels, history
    if not history:
        print("Nothing to undo.")
        return
    removed = history.pop()
    filled_pixels.difference_update(removed)
    redraw_filled_pixels()
    print("Done. undo last fill.")
    cv2.imshow("Flood Fill (1000x500)", img)

# -------------------------------------------
# Helper: Bresenham's Line Algorithm
# Computes a list of pixels along a line.
# The input coordinates are in top-left system,
# and we convert each to bottom-left before returning.
# -------------------------------------------
def bresenham_line(x0, y0, x1, y1):
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    pixels = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    if dx > dy:
        err = dx // 2
        while x != x1:
            # Convert to bottom-left coordinate
            pixels.append((x, (h - 1) - y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
        pixels.append((x1, (h - 1) - y1))
    else:
        err = dy // 2
        while y != y1:
            pixels.append((x, (h - 1) - y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
        pixels.append((x1, (h - 1) - y1))
    return pixels

# -------------------------------------------
# Mode Switching Variables and States
# -------------------------------------------
# mode can be "bucket", "line", or "paint"
mode = "bucket"  # default mode
print("Current mode: Bucket Fill (press 'l' for line, 'b' for bucket, 'p' for paint)")

# Variables for line mode:
line_start = None
is_drawing_line = False

# Variables for paint mode:
is_painting = False
current_paint_pixels = set()
is_flooding = False

# -------------------------------------------
# Mouse Callback
# -------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global mode, is_flooding, is_drawing_line, line_start, is_painting, current_paint_pixels, img, filled_pixels, history

    if mode == "bucket":
        # Bucket Fill Mode: Trigger flood fill on left click.
        if event == cv2.EVENT_LBUTTONDOWN:
            is_flooding = True
        elif event == cv2.EVENT_MOUSEMOVE and is_flooding:
            flood_fill(x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            is_flooding = False
    elif mode == "line":
        # Line Mode: Draw a line by dragging.
        if event == cv2.EVENT_LBUTTONDOWN:
            line_start = (x, y)
            is_drawing_line = True
        elif event == cv2.EVENT_MOUSEMOVE and is_drawing_line:
            temp_img = img.copy()
            cv2.line(temp_img, line_start, (x, y), (0, 255, 0), thickness=1)
            cv2.imshow("Flood Fill (1000x500)", temp_img)
        elif event == cv2.EVENT_LBUTTONUP and is_drawing_line:
            is_drawing_line = False
            # Draw final line
            cv2.line(img, line_start, (x, y), (0, 255, 0), thickness=1)
            cv2.imshow("Flood Fill (1000x500)", img)
            # Compute line pixels using Bresenham algorithm and update filled_pixels
            line_pixels = bresenham_line(line_start[0], line_start[1], x, y)
            filled_pixels.update(line_pixels)
            history.append(set(line_pixels))
            print(f"Line drawn. Total filled pixels: {len(filled_pixels)}")

    elif mode == "paint":
        # Paint Mode: Freehand drawing with a brush.
        brush_radius = 2
        if event == cv2.EVENT_LBUTTONDOWN:
            is_painting = True
            current_paint_pixels = set()
        elif event == cv2.EVENT_MOUSEMOVE and is_painting:
            # Draw a filled circle at the current mouse location.
            cv2.circle(img, (x, y), brush_radius, (0, 255, 0), -1)
            cv2.imshow("Flood Fill (1000x500)", img)
            # Add all points within the brush radius to current_paint_pixels.
            for dx in range(-brush_radius, brush_radius + 1):
                for dy in range(-brush_radius, brush_radius + 1):
                    if dx*dx + dy*dy <= brush_radius*brush_radius:
                        px = x + dx
                        py = y + dy
                        # Convert to bottom-left coordinates
                        current_paint_pixels.add((px, (h - 1) - py))
                        filled_pixels.add((px, (h - 1) - py))
        elif event == cv2.EVENT_LBUTTONUP:
            is_painting = False
            if current_paint_pixels:
                history.append(current_paint_pixels)
            print(f"Painted. Total filled pixels: {len(filled_pixels)}")

cv2.setMouseCallback("Flood Fill (1000x500)", mouse_callback)

# -------------------------------------------
# Main Loop
# -------------------------------------------
while True:
    key = cv2.waitKey(1) & 0xFF
    # Press 'l' for Line Mode, 'b' for Bucket Fill Mode, 'p' for Paint Mode.
    if key == ord('l'):
        mode = "line"
        print("Switched to Line Mode")
    elif key == ord('b'):
        mode = "bucket"
        print("Switched to Bucket Fill Mode")
    elif key == ord('p'):
        mode = "paint"
        print("Switched to Paint Mode")
    elif key == ord('q'):
        # Save filled pixels to JSON on quit.
        with open(json_filename, 'w') as f:
            json.dump([[int(v) for v in px] for px in filled_pixels], f)
        print(f"Saved {len(filled_pixels)} pixels to {json_filename}")
        break
    elif key == ord('z'):
        undo_last_fill()

cv2.destroyAllWindows()
