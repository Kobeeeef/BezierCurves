import cv2
import numpy as np
import json
import os

#############################################
# Flood Fill with Bottom-Left Coordinates,
# NO RESIZING -- uses full original resolution.
#############################################

json_filename = "filled_pixels.json"  # Where to save your fill coordinates
image_path = "2025game-field.jpg"     # Change to your path

# Load full-resolution image
img_original = cv2.imread(image_path)
if img_original is None:
    raise ValueError("Could not load image! Check file path.")

# Dimensions of the original image
h, w = img_original.shape[:2]

# We'll use this for on-screen display (no resize)
img = img_original.copy()

# -------------------------------------------
# Global Variables
# -------------------------------------------
filled_pixels = set()  # set of (x, y) in bottom-left coords
history = []           # stack of sets for undo
is_drawing = False     # track mouse dragging

# -------------------------------------------
# Load existing fill data from JSON
# -------------------------------------------
if os.path.exists(json_filename):
    try:
        with open(json_filename, "r") as f:
            loaded = json.load(f)
            filled_pixels = set(tuple(p) for p in loaded)
            print(f"Loaded {len(filled_pixels)} pixels from {json_filename}")
    except (json.JSONDecodeError, ValueError):
        print("JSON file invalid or empty. Starting fresh.")

# -------------------------------------------
# Helper: Redraw existing fills
# -------------------------------------------
def redraw_filled_pixels():
    global img
    # Reset to original
    img = img_original.copy()
    for (lx, ly) in filled_pixels:
        # Convert bottom-left coords -> top-left coords
        col = lx
        row = (h - 1) - ly
        if 0 <= col < w and 0 <= row < h:
            img[row, col] = (0, 255, 0)

redraw_filled_pixels()
cv2.imshow("Flood Fill (No Resize)", img)

# -------------------------------------------
# Flood Fill Operation
# -------------------------------------------
def flood_fill(start_x, start_y):
    global filled_pixels, history, img

    # Copy for flood fill
    img_copy = img.copy()
    # For floodFill, we need a mask with a 1-pixel border
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    loDiff = (10, 10, 10)
    upDiff = (10, 10, 10)
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

    # Remove the 1-pixel border
    mask = mask[1:-1, 1:-1]

    # Find filled areas
    filled_locs = np.column_stack(np.where(mask > 0))  # (rows, cols)
    new_pixels = set()
    for row, col in filled_locs:
        # Convert from top-left coords to bottom-left
        local_x = col
        local_y = (h - 1) - row
        new_pixels.add((local_x, local_y))

    if new_pixels:
        history.append(new_pixels)
        filled_pixels.update(new_pixels)
        img = img_copy
        cv2.imshow("Flood Fill (No Resize)", img)
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
    cv2.imshow("Flood Fill (No Resize)", img)

# -------------------------------------------
# Mouse Callback
# -------------------------------------------
def mouse_callback(event, x, y, flags, param):
    global is_drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        flood_fill(x, y)
    elif event == cv2.EVENT_MOUSEMOVE and is_drawing:
        flood_fill(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False

cv2.setMouseCallback("Flood Fill (No Resize)", mouse_callback)

# -------------------------------------------
# Main Loop
# -------------------------------------------
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Save
        with open(json_filename, 'w') as f:
            json.dump([[int(v) for v in px] for px in filled_pixels], f)
        print(f"Saved {len(filled_pixels)} pixels to {json_filename}")
        break
    elif key == ord('z'):
        undo_last_fill()

cv2.destroyAllWindows()
