import cv2
import numpy as np
import json
import os
import screeninfo

# Define ROI boundaries (only work within these bounds)
bottomBoundary = 91
topBoundary = 1437
leftBoundary = 421
rightBoundary = 3352

# Load the original image
img_original = cv2.imread("./2025field.png")  # Change to your image file
if img_original is None:
    raise ValueError("Image not found!")
h, w = img_original.shape[:2]

# Get screen size
screen = screeninfo.get_monitors()[0]
screen_w, screen_h = screen.width, screen.height

# Resize image to fit the screen (without upscaling)
scale_w = screen_w / w
scale_h = screen_h / h
scale = min(scale_w, scale_h, 1.0)
new_w, new_h = int(w * scale), int(h * scale)

img = cv2.resize(img_original, (new_w, new_h))
h, w = img.shape[:2]  # Update dimensions after resizing

# Global variables
filled_pixels = set()  # Will store filled pixels as coordinates relative to the ROI
history = []          # For undo (stores sets of newly filled relative pixels)
is_drawing = False    # Tracks if the mouse is being dragged
json_filename = "static_obstacles_field.json"

# Load previous filled pixels (relative coordinates) if available
if os.path.exists(json_filename):
    with open(json_filename, "r") as f:
        try:
            loaded_pixels = json.load(f)
            filled_pixels = set(tuple(p) for p in loaded_pixels)
            print(f"Loaded {len(filled_pixels)} pixels from {json_filename}")
        except json.JSONDecodeError:
            print("Error loading JSON file. Starting fresh.")

# Redraw loaded pixels onto the image by converting relative ROI coords to absolute image coords.
for rel_x, rel_y in filled_pixels:
    absolute_x = rel_x + leftBoundary
    absolute_y = rel_y + bottomBoundary
    # Check boundaries in case of rounding/scaling differences
    if 0 <= absolute_x < w and 0 <= absolute_y < h:
        img[absolute_y, absolute_x] = (0, 255, 0)

def flood_fill(x, y):
    """Apply flood fill from the given point (x, y) and extract filled pixels within the ROI.
    The stored pixels will be relative to the ROI.
    """
    global filled_pixels, img, history

    # Only allow flood fill if the starting point is within the ROI.
    if not (leftBoundary <= x < rightBoundary and bottomBoundary <= y < topBoundary):
        print("Clicked point is outside the allowed ROI.")
        return

    img_copy = img.copy()
    # Create mask with a 1-pixel border (required by floodFill)
    mask = np.zeros((h + 2, w + 2), np.uint8)
    loDiff = upDiff = (10, 10, 10, 10)  # Color tolerance
    flood_fill_color = (0, 255, 0)       # Fill color

    # Perform the flood fill
    cv2.floodFill(img_copy, mask, (x, y), flood_fill_color, loDiff, upDiff, flags=cv2.FLOODFILL_FIXED_RANGE)

    # Remove the border from the mask
    mask = mask[1:-1, 1:-1]
    # Find all filled (nonzero) pixels
    nonzero_pixels = np.column_stack(np.where(mask > 0))

    new_pixels = set()
    for py, px in nonzero_pixels:
        x_val = int(px)
        y_val = int(py)
        # Only accept points within the ROI
        if leftBoundary <= x_val < rightBoundary and bottomBoundary <= y_val < topBoundary:
            # Convert absolute image coordinate to relative ROI coordinate
            new_pixels.add((x_val - leftBoundary, y_val - bottomBoundary))

    if new_pixels:
        history.append(new_pixels)  # Save history for undo
        filled_pixels.update(new_pixels)

    print(f"Total filled pixels so far: {len(filled_pixels)}")
    img[:] = img_copy[:]  # Update the displayed image
    cv2.imshow("Click or Drag to Fill", img)

def undo_last_fill():
    """Undo the last filled region."""
    global filled_pixels, history, img

    if history:
        last_filled = history.pop()
        filled_pixels.difference_update(last_filled)

        # Reset the image to the original resized image.
        img[:] = cv2.resize(img_original, (new_w, new_h))[:]
        # Redraw the remaining filled pixels (convert relative back to absolute)
        for rel_x, rel_y in filled_pixels:
            absolute_x = rel_x + leftBoundary
            absolute_y = rel_y + bottomBoundary
            if 0 <= absolute_x < w and 0 <= absolute_y < h:
                img[absolute_y, absolute_x] = (0, 255, 0)

        print(f"Undo complete! Remaining filled pixels: {len(filled_pixels)}")
        cv2.imshow("Click or Drag to Fill", img)

def mouse_callback(event, x, y, flags, param):
    global is_drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        flood_fill(x, y)
    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            flood_fill(x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False

# Display image and set the mouse callback
cv2.imshow("Click or Drag to Fill", img)
cv2.setMouseCallback("Click or Drag to Fill", mouse_callback)

# Main loop: quit when 'q' is pressed, or undo last fill when 'z' is pressed.
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Saving pixels to JSON...")
        with open(json_filename, "w") as f:
            json.dump(list(filled_pixels), f)
        print(f"Saved {len(filled_pixels)} pixels to {json_filename}")
        break
    elif key == ord('z'):
        undo_last_fill()

cv2.destroyAllWindows()
