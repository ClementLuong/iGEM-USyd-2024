import cv2
import numpy as np

# Parameters
SCALE_BAR_PIXELS = 124  # Length of scale bar in pixels
SCALE_BAR_MICROMETERS = 20  # Length of scale bar in micrometers
PARTICLE_MIN_LENGTH_MICROMETERS = 2  # Minimum length of particles to keep
PARTICLE_MAX_LENGTH_MICROMETERS = 15  # Maximum length of particles to keep
MIN_ASPECT_RATIO = 1.5  # Minimum length-to-width ratio for rod-like features

# Calculate pixel-to-micrometer conversion
pixel_to_micrometer = SCALE_BAR_MICROMETERS / SCALE_BAR_PIXELS
min_length_pixels = PARTICLE_MIN_LENGTH_MICROMETERS / pixel_to_micrometer
max_length_pixels = PARTICLE_MAX_LENGTH_MICROMETERS / pixel_to_micrometer

def enhance_and_process_image(image):
    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    cv2.imwrite("contrast_enhanced.tif", enhanced)

    # Normalize intensities to the full 0-255 range
    normalized = cv2.normalize(enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite("normalized_image.tif", normalized)

    # Blurring to reduce noise
    blurred = cv2.GaussianBlur(normalized, (7, 7), 0)
    cv2.imwrite("blurred_image.tif", blurred)

    # Adaptive thresholding to handle varying intensity across the image
    thresholded = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 11, 2)
    cv2.imwrite("adaptive_thresholded_image.tif", thresholded)

    # Invert the thresholded image
    inverted = cv2.bitwise_not(thresholded)
    cv2.imwrite("inverted_image.tif", inverted)

    # Apply morphological opening (erosion followed by dilation) to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    cv2.imwrite("opened_image.tif", opened)

    # Re-invert the image after dilation
    final_dilated = cv2.bitwise_not(opened)
    cv2.imwrite("final_dilated_image.tif", final_dilated)
    return final_dilated

def get_contours(image):
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area (adjust the threshold as needed)
    # filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 50]

    # Output image for visualization
    output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Draw contours on the final image
    debug = output.copy()
    cv2.drawContours(debug, contours, -1, (255, 0, 0), 1)
    cv2.imwrite("all_detected_contours.tif", debug)
    return contours

# Function to calculate the aspect ratio of the best fitted bounding box
def get_min_box(contour, scaling_factor):
    # Get the rotated bounding box using minAreaRect
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)  # Get 4 vertices of the rotated box
    width = cv2.norm(box[0] - box[1])  # Distance between two points (width)
    height = cv2.norm(box[1] - box[2])  # Distance between two points (height)

    # Calculate aspect ratio (long/short side of the bounding box)
    aspect_ratio = max(width, height) / min(width, height) if min(width, height) != 0 else 0

    # Calculate the physical length of the particle (in microns)
    length_in_microns = max(width, height) * scaling_factor

    return aspect_ratio, length_in_microns

# Function to calculate solidity of the contour
def get_solidity(contour):
    # Convex hull
    hull = cv2.convexHull(contour)
    area = cv2.contourArea(contour)
    hull_area = cv2.contourArea(hull)
    return float(area) / hull_area if hull_area != 0 else 0

# Function to calculate circularity of a contour
def get_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:  # Avoid division by zero
        return 0
    circularity = (4 * np.pi * area) / (perimeter ** 2)
    return circularity

# Load image
image = cv2.imread("Data/Test.tif", cv2.IMREAD_UNCHANGED)  # Load .tif image
if image is None:
    raise ValueError("Failed to load image. Check the file path and format.")

# Convert to grayscale if necessary
if len(image.shape) == 3:  # Check if the image is multi-channel
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray = image

enhanced = enhance_and_process_image(gray)
contours = get_contours(enhanced)

# Process each contour
filtered_contours = []
for contour in contours:
    aspect_ratio, length_in_microns = get_min_box(contour, pixel_to_micrometer)
    solidity = get_solidity(contour)
    circularity = get_circularity(contour)

    if aspect_ratio > MIN_ASPECT_RATIO and solidity > 0.6 and PARTICLE_MIN_LENGTH_MICROMETERS <= length_in_microns <= PARTICLE_MAX_LENGTH_MICROMETERS:
        filtered_contours.append(contour)

# Draw filtered contours on the image
output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cv2.drawContours(output, filtered_contours, -1, (0, 255, 0), 2)
cv2.imwrite("filtered_contours_with_size.tif", output)

# Display the result
cv2.imshow("Filtered Contours with Size", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # Preprocessing
# blurred = cv2.GaussianBlur(binary, (7, 7), 0)
# edges = cv2.Canny(blurred, 100, 200)