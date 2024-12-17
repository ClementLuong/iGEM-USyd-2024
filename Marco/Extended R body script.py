import cv2
import numpy as np
import math

# Parameters
SCALE_BAR_PIXELS = 124  # Length of scale bar in pixels
SCALE_BAR_MICROMETERS = 20  # Length of scale bar in micrometers
PARTICLE_MIN_LENGTH_MICROMETERS = 2  # Minimum length of particles to keep
PARTICLE_MAX_LENGTH_MICROMETERS = 15  # Maximum length of particles to keep
MIN_ASPECT_RATIO = 3  # Minimum length-to-width ratio for rod-like features

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

# For contamination
def enhance_and_process_image_cont(image):
    # Contrast enhancement using CLAHE
    clahe_cont = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_cont = clahe_cont.apply(image)
    cv2.imwrite("contrast_enhanced_cont.tif", enhanced_cont)

    # Normalize intensities to the full 0-255 range
    normalized_cont = cv2.normalize(enhanced_cont, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    cv2.imwrite("normalized_cont_image.tif", normalized_cont)

    # Blurring to reduce noise
    blurred_cont = cv2.GaussianBlur(normalized_cont, (7, 7), 0)
    cv2.imwrite("blurred_cont_image.tif", blurred_cont)

    return blurred_cont

def get_contours(image):
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

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

# Filter for contamination
def filter_for_contamination(image):
    # Threshold to detect dark regions (contaminants)
    _, dark_thresholded = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite("dark_thresholded_for_contamination.tif", dark_thresholded)

    # Threshold to detect light regions (contaminants)
    _, light_thresholded = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite("light_thresholded_for_contamination.tif", light_thresholded)

    # Combine both dark and light regions to create a contamination mask
    contamination_mask = cv2.bitwise_or(dark_thresholded, light_thresholded)
    cv2.imwrite("combined_contamination_mask.tif", contamination_mask)

    # Find contours in the combined mask (dark + light regions)
    contours = get_contours(contamination_mask)

    # Filter contours based on area and size if needed
    filtered_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:  # Adjust area threshold if necessary
            filtered_contours.append(contour)

    return filtered_contours

def resample_contour(contour, num_samples):
    # Calculate the cumulative distance (arc length) of points along the contour
    dist = np.cumsum(np.sqrt(np.sum(np.diff(contour, axis=0)**2, axis=1)))
    dist = np.insert(dist, 0, 0)  # Insert 0 at the start for cumulative distance

    # Normalize distances
    total_distance = dist[-1]
    normalized_distances = dist / total_distance

    # Create an array of evenly spaced positions along the contour
    target_distances = np.linspace(0, 1, num_samples)

    # Interpolate points along the contour based on the target distances
    resampled_contour = np.zeros((num_samples, 2))
    for i in range(2):  # Iterate over x and y coordinates
        resampled_contour[:, i] = np.interp(target_distances, normalized_distances, contour[:, i])

    return resampled_contour

# Optimized helper function to calculate approximate average distance
def approx_average_distance_between_contours(contour1, contour2, num_samples=100):
    # Convert contours to NumPy arrays and flatten them
    points1 = np.squeeze(np.array(contour1), axis=1)
    points2 = np.squeeze(np.array(contour2), axis=1)

    # Resample contours to get a more uniform distribution of points along the contours
    # Use interpolation (e.g., linear or arc length-based resampling)
    contour1_resampled = resample_contour(points1, num_samples)
    contour2_resampled = resample_contour(points2, num_samples)

    # Compute pairwise distances between the resampled points
    distances = np.linalg.norm(contour1_resampled[:, None] - contour2_resampled, axis=2)

    # Calculate the average distance as the mean of all pairwise distances
    average_distance = np.mean(distances)

    return average_distance

def angle_of_line(x1, y1, x2, y2):
    return abs(math.atan2(y2 - y1, x2 - x1))

def point_to_line_distance(px, py, x1, y1, x2, y2):
    line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
    if line_length_sq == 0:  # Line is a point
        return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)

    # Project point onto the line segment and clamp t to [0, 1]
    t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
    proj_x = x1 + t * (x2 - x1)
    proj_y = y1 + t * (y2 - y1)

    # Return the Euclidean distance from the point to the projection
    return math.sqrt((px - proj_x) ** 2 + (py - proj_y) ** 2)

def line_to_line_distance(x1, y1, x2, y2, x3, y3, x4, y4):
    # Calculate distances between endpoints of one line and the other line
    distances = [
        point_to_line_distance(x1, y1, x3, y3, x4, y4),
        point_to_line_distance(x2, y2, x3, y3, x4, y4),
        point_to_line_distance(x3, y3, x1, y1, x2, y2),
        point_to_line_distance(x4, y4, x1, y1, x2, y2)
    ]
    return min(distances)

def reduce_redundant_lines(lines, angle_threshold=15, distance_threshold=15):
    # Convert angle threshold to radians
    angle_threshold = math.radians(angle_threshold)

    # Store each line with its angle and midpoint
    line_data = []
    for line in lines:
        x1, y1, x2, y2 = line
        angle = angle_of_line(x1, y1, x2, y2)
        midpoint = ((x1 + x2) / 2, (y1 + y2) / 2)  # Ensure midpoint is a tuple
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        line_data.append((line, angle, midpoint, length))

    # Group lines into clusters
    clusters = []
    for i, (line, angle, midpoint, length) in enumerate(line_data):
        x1, y1, x2, y2 = line
        added_to_cluster = False
        for cluster in clusters:
            # Check if the line is parallel and close to any line in the cluster
            if all(abs(angle - other_angle) < angle_threshold for _, other_angle, _, _ in cluster):
                if all(
                    line_to_line_distance(x1, y1, x2, y2, other_line[0], other_line[1], other_line[2], other_line[3]) < distance_threshold
                    for other_line, _, _, _ in cluster
                ):
                    cluster.append((line, angle, midpoint, length))
                    added_to_cluster = True
                    break

        if not added_to_cluster:
            clusters.append([(line, angle, midpoint, length)])

    reduced_lines = []
    for cluster in clusters:
        # Find the line that most closely traces the contour
        def line_tracing_score(line):
            x1, y1, x2, y2 = map(float, line)
            # Calculate distances to the contour for the endpoints and midpoint
            distances = [
                cv2.pointPolygonTest(contour, (x1, y1), True),
                cv2.pointPolygonTest(contour, (x2, y2), True),
                cv2.pointPolygonTest(contour, ((x1 + x2) / 2, (y1 + y2) / 2), True),
            ]
            # Use the sum of absolute distances as the score (lower is better)
            return sum(abs(d) for d in distances)

        # Select the line with the lowest tracing score
        representative = min(cluster, key=lambda x: line_tracing_score(x[0]))
        reduced_lines.append(representative[0])

    return reduced_lines


# Load image
image = cv2.imread("/Users/marco/Desktop/Images for python script/1_50_pH5_1-Image Export-01/1_50_pH5_1-Image Export-01.jpg", cv2.IMREAD_UNCHANGED)  # Load .tif image
if image is None:
    raise ValueError("Failed to load image. Check the file path and format.")

# Convert to grayscale if necessary
if len(image.shape) == 3:  # Check if the image is multi-channel
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray = image

enhanced = enhance_and_process_image(gray)
contours = get_contours(enhanced)

blurred_cont = enhance_and_process_image_cont(gray)
cont_contours = filter_for_contamination(blurred_cont)

# Process each contour
filtered_contours = []
remaining_contours = []
for contour in contours:
    aspect_ratio, length_in_microns = get_min_box(contour, pixel_to_micrometer)
    solidity = get_solidity(contour)

    if aspect_ratio > 2 and solidity > 0.1 and PARTICLE_MIN_LENGTH_MICROMETERS <= length_in_microns <= PARTICLE_MAX_LENGTH_MICROMETERS:
        filtered_contours.append(contour)
    elif 200 < cv2.contourArea(contour) < 50000:
        is_far_enough = all(
            approx_average_distance_between_contours(contour, cont_contour) > 20 for cont_contour in cont_contours
        )
        if is_far_enough:
            remaining_contours.append(contour)

# Detect edges
edges = cv2.Canny(enhanced, 50, 150)

# Constrain line segments to individual contours
output = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
line_lengths = []
for contour in filtered_contours:
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    contour_edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Fit lines to points within the contour
    points = np.vstack(contour).squeeze()
    if points.shape[0] >= 2:  # Ensure there are enough points to fit a line
        [vx, vy, x, y] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)

        # Calculate the start and end points of the line segment within the contour
        x_min, y_min, w, h = cv2.boundingRect(contour)
        x_max, y_max = x_min + w, y_min + h

        t0 = ((x_min - x) / vx) if vx != 0 else float('-inf')
        t1 = ((x_max - x) / vx) if vx != 0 else float('inf')
        t2 = ((y_min - y) / vy) if vy != 0 else float('-inf')
        t3 = ((y_max - y) / vy) if vy != 0 else float('inf')

        t_start = max(min(t0, t1), min(t2, t3))
        t_end = min(max(t0, t1), max(t2, t3))

        x_start = int((x + t_start * vx).item())
        y_start = int((y + t_start * vy).item())
        x_end = int((x + t_end * vx).item())
        y_end = int((y + t_end * vy).item())

        # Add length to line length vector
        line_lengths.append(math.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2)*pixel_to_micrometer)

        # Draw the line segment
        cv2.line(output, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

Hough_lines = 0
for contour in remaining_contours:
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
    contour_edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Apply Hough Transform to detect multiple lines in the contour region
    lines = cv2.HoughLinesP(contour_edges, 1, np.pi / 180, threshold=15, minLineLength=min_length_pixels, maxLineGap=50)

    if lines is not None:
        # Convert lines to a simpler format for processing
        lines_list = [(line[0][0], line[0][1], line[0][2], line[0][3]) for line in lines]

        # Reduce redundant lines
        reduced_lines = reduce_redundant_lines(lines_list)

        # Draw the reduced lines
        for line in reduced_lines:
            x1, y1, x2, y2 = line

            # Constrain the lines within the bounding box of the contour
            x_min, y_min, w, h = cv2.boundingRect(contour)
            x_max, y_max = x_min + w, y_min + h

            # Ensure the line is drawn within the bounding box
            x1 = max(x_min, min(x1, x_max))
            y1 = max(y_min, min(y1, y_max))
            x2 = max(x_min, min(x2, x_max))
            y2 = max(y_min, min(y2, y_max))

            # Draw the line on the image
            cv2.line(output, (x1, y1), (x2, y2), (125, 130, 0), 2)

            Hough_lines += 1

cv2.imwrite("Extended_no_open.tif", output)

# Draw filtered contours on the image
# cv2.drawContours(output, cont_contours, -1, (255, 0, 0), 2)  # Green color for contamination contours
# cv2.drawContours(output, remaining_contours, -1, (0, 0, 255), 2)  # Show remaining contours
cv2.imwrite("close_to_contour.tif", output)

#Average length of R bodies below
Average_ext_length = round(sum(line_lengths) / len(line_lengths), 2)

#Number of extended R bodies
Number_ext_R_bodies = len(line_lengths) + Hough_lines

print(f"Average R body length: {Average_ext_length} μm")
print(f"Number of extended R bodies {Number_ext_R_bodies}")

# Display the result
cv2.imshow("Final", output)
cv2.waitKey(0)
cv2.destroyAllWindows()
