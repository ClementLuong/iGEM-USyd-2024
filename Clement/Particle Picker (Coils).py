import cv2
import numpy as np

def isolate_high_contrast_dots(input_path, output_path, contrast_threshold=1, debug_folder="debug"):
    import os

    # Create a debug folder if it doesn't exist
    os.makedirs(debug_folder, exist_ok=True)

    # Load the original image
    image = cv2.imread(input_path)
    if image is None:
        print(f"Error: Unable to load the image at {input_path}")
        return
    cv2.imwrite(f"{debug_folder}/01_original.tif", image)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{debug_folder}/02_grayscale.tif", gray)

    # Apply a median blur to reduce noise
    blurred = cv2.medianBlur(gray, 5)
    cv2.imwrite(f"{debug_folder}/03_blurred.tif", blurred)

    # Use adaptive thresholding to handle uneven lighting
    binary = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    cv2.imwrite(f"{debug_folder}/04_adaptive_threshold.tif", binary)

    # Perform morphological operations to reduce noise further
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary_cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    cv2.imwrite(f"{debug_folder}/05_binary_cleaned.tif", binary_cleaned)

    # Find contours
    contours, _ = cv2.findContours(binary_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Copy the original image to overlay dots
    overlay_image = image.copy()

    # Create a mask to visualize filtered contours
    mask_filtered = np.zeros_like(gray)

    # Filter contours based on size, shape, and contrast
    for contour in contours:
        # Calculate area and circularity
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))

        # Skip contours that don't meet size and shape criteria
        if not (1 < area < 500 and 0.5 < circularity <= 1.2):
            continue

        # Create a mask for the current contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)

        # Calculate average intensity inside and around the contour
        contour_mean = cv2.mean(gray, mask)[0]

        # Dilate the mask to include surrounding area
        dilated_mask = cv2.dilate(mask, None, iterations=5)
        surrounding_mask = cv2.bitwise_and(dilated_mask, cv2.bitwise_not(mask))
        surrounding_mean = cv2.mean(gray, surrounding_mask)[0]

        # Calculate contrast difference
        contrast_difference = abs(contour_mean - surrounding_mean)

        # Keep only high-contrast particles
        if contrast_difference >= contrast_threshold:
            cv2.drawContours(overlay_image, [contour], -1, (0, 0, 255), 2)
            cv2.drawContours(mask_filtered, [contour], -1, 255, -1)

    # Save debugging and final output
    cv2.imwrite(f"{debug_folder}/06_mask_filtered.tif", mask_filtered)
    cv2.imwrite(f"{debug_folder}/07_overlay_image.tif", overlay_image)
    cv2.imwrite(output_path, overlay_image)
    print(f"Image with high-contrast dots saved to {output_path}")

    # Display the overlaid image
    cv2.imshow("High-Contrast Dots", overlay_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage example
input_image_path = "Data/Coiled.png"
output_image_path = "output_high_contrast.tif"
isolate_high_contrast_dots(input_image_path, output_image_path, contrast_threshold=20, debug_folder="debug")
