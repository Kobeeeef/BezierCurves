import cv2
import numpy as np
import json
import os
import screeninfo

# Load the image
img_original = cv2.imread("./2025field.png")  # Change to your image file
h, w = img_original.shape[:2]

# Get screen size
screen = screeninfo.get_monitors()[0]
screen_w, screen_h = screen.width, screen.height

# Resize image to fit screen
scale_w = screen_w / w
scale_h = screen_h / h
scale = min(scale_w, scale_h, 1.0)  # Ensure we don't upscale
new_w, new_h = int(w * scale), int(h * scale)

img = cv2.resize(img_original, (new_w, new_h))
h, w = img.shape[:2]  # Update dimensions after resizing

# Global variables
filled_pixels = set()  # Store all filled pixels
history = []  # Stores history for undo (Z key)
is_drawing = False  # Tracks if the mouse is being dragged
json_filename = "filled_pixels.json"

# Load previous filled pixels if available
if os.path.exists(json_filename):
    with open(json_filename, "r") as f:
        try:
            loaded_pixels = json.load(f)
            filled_pixels = set(tuple(p) for p in loaded_pixels)
            print(f"Loaded {len(filled_pixels)} pixels from {json_filename}")
        except json.JSONDecodeError:
            print("Error loading JSON file. Starting fresh.")

# Redraw loaded pixels onto the image
for px, py in filled_pixels:
    img[py, px] = (0, 255, 0)  # Restore previous fills

def flood_fill(x, y):
    """Apply flood fill from the given point (x, y) and extract filled pixels."""
    global filled_pixels, img, history

    img_copy = img.copy()
    mask = np.zeros((h+2, w+2), np.uint8)  # Mask for flood fill
    loDiff = upDiff = (20, 20, 20, 20)  # Color tolerance
    flood_fill_color = (0, 255, 0)  # Visualization color

    # Apply flood fill
    cv2.floodFill(img_copy, mask, (x, y), flood_fill_color, loDiff, upDiff, flags=cv2.FLOODFILL_FIXED_RANGE)

    # Optimize extraction using NumPy
    mask = mask[1:-1, 1:-1]  # Remove flood fill border
    nonzero_pixels = np.column_stack(np.where(mask > 0))  # Extract non-zero pixels

    # Convert to a set of (x, y) coordinates
    new_pixels = set((int(px), int(py)) for py, px in nonzero_pixels)

    if new_pixels:
        history.append(new_pixels)  # Save history for undo
        filled_pixels.update(new_pixels)  # Store pixels in global set

    print(f"Total filled pixels so far: {len(filled_pixels)}")

    # Update the displayed image
    img[:] = img_copy[:]  # Apply changes to the main image
    cv2.imshow("Click or Drag to Fill", img)

def undo_last_fill():
    """Removes the last filled region."""
    global filled_pixels, history, img

    if history:
        last_filled = history.pop()  # Get the last filled region
        filled_pixels.difference_update(last_filled)  # Remove those pixels

        # Reset the image and redraw remaining fills
        img[:] = cv2.resize(img_original, (new_w, new_h))[:]  # Reset to original
        for px, py in filled_pixels:
            img[py, px] = (0, 255, 0)  # Redraw previous fills

        print(f"Undo complete! Remaining filled pixels: {len(filled_pixels)}")
        cv2.imshow("Click or Drag to Fill", img)

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global is_drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        is_drawing = True
        flood_fill(x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if is_drawing:
            flood_fill(x, y)  # Keep filling while dragging

    elif event == cv2.EVENT_LBUTTONUP:
        is_drawing = False  # Stop filling when released

# Display image and set mouse callback
cv2.imshow("Click or Drag to Fill", img)
cv2.setMouseCallback("Click or Drag to Fill", mouse_callback)

# Loop until 'Q' is pressed
while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Quit when 'Q' is pressed
        print("Saving pixels to JSON...")
        with open(json_filename, "w") as f:
            json.dump(list(filled_pixels), f)
        print(f"Saved {len(filled_pixels)} pixels to {json_filename}")
        break
    elif key == ord('z'):  # Undo last fill when 'Z' is pressed
        undo_last_fill()

cv2.destroyAllWindows()
