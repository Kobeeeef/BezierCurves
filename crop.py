import cv2


def crop_image(image_path, output_path):
    # Define boundaries
    bottomBoundary = 91  # lowest Y value in the original image
    topBoundary = 1437  # highest Y value in the original image
    leftBoundary = 421  # leftmost X value in the original image
    rightBoundary = 3352  # rightmost X value in the original image

    # Read the image
    image = cv2.imread(image_path)

    # Check if image is loaded
    if image is None:
        print("Error: Unable to load image.")
        return

    # Crop the image
    cropped_image = image[bottomBoundary:topBoundary, leftBoundary:rightBoundary]

    # Save the cropped image
    cv2.imwrite(output_path, cropped_image)
    print(f"Cropped image saved to {output_path}")


# Example usage
input_image = "2025field.png"  # Change to your input image path
output_image = "2025game-field.jpg"  # Change to your desired output path
crop_image(input_image, output_image)
