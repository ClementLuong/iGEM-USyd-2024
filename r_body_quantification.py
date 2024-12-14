import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
import os
import pandas as pd
import csv

# Parameters
SCALE_BAR_PIXELS = 124  # Length of scale bar in pixels
SCALE_BAR_MICROMETERS = 20  # Length of scale bar in micrometers
PARTICLE_MIN_LENGTH_MICROMETERS = 2  # Minimum length of particles to keep
PARTICLE_MAX_LENGTH_MICROMETERS = 20  # Maximum length of particles to keep
MIN_ASPECT_RATIO = 3  # Minimum length-to-width ratio for rod-like features

# Calculate pixel-to-micrometer conversion
pixel_to_micrometer = SCALE_BAR_MICROMETERS / SCALE_BAR_PIXELS
min_length_pixels = PARTICLE_MIN_LENGTH_MICROMETERS / pixel_to_micrometer
max_length_pixels = PARTICLE_MAX_LENGTH_MICROMETERS / pixel_to_micrometer

#--------------EXTENDED R BODIES-------------------------------


def enhance_and_process_image(image):
    # Contrast enhancement using CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(20, 20))
    enhanced = clahe.apply(image)
    # cv2.imwrite("contrast_enhanced.tif", enhanced)

    # Normalize intensities to the full 0-255 range
    normalized = cv2.normalize(enhanced, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    # cv2.imwrite("normalized_image.tif", normalized)

    # Median Blurring to reduce noise
    m_blurred = cv2.medianBlur(normalized, 5)
    # cv2.imwrite("m_blurred_image.tif", m_blurred)

    # Adaptive thresholding to handle varying intensity across the image
    thresholded = cv2.adaptiveThreshold(m_blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                 cv2.THRESH_BINARY, 31, 2)
      
    # cv2.imwrite("adaptive_thresholded_image.tif", thresholded)

    # Invert the thresholded image
    inverted = cv2.bitwise_not(thresholded)
    # cv2.imwrite("inverted_image.tif", inverted)

    # Apply morphological opening (erosion followed by dilation) to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    opened = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    # cv2.imwrite("opened_image.tif", opened)

    # Re-invert the image after dilation
    final_dilated = cv2.bitwise_not(opened)
    # cv2.imwrite("final_dilated_image.tif", final_dilated)
    return final_dilated

def get_contours(image):

    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area (adjust the threshold as needed)
    # filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 50]

    # Output image for visualization
    output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Draw contours on the final image
    debug = output.copy()
    cv2.drawContours(debug, contours, -1, (255, 0, 0), 1)
    # cv2.imwrite("all_detected_contours.tif", debug)
    return contours

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

    # Calculate extent of contour
    box_area = width*height

    extent = cv2.contourArea(contour)/box_area if box_area != 0 else 0

    return aspect_ratio, length_in_microns, extent


# Function for handling extended r bodies. Calls the previous functions. Returns the number of exteded R-bodies and the contours for them.  
def get_extended(file):
        # Load image
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)  # Load .tif image. need to append to parent directory
    if image is None:
        raise ValueError("Failed to load image. Check the file path and format.")

    enhanced = enhance_and_process_image(image) #returns a thresholded image, used to get contours. 

    contours = get_contours(enhanced) # returns all contours. 

    # Process each contour
    filtered_contours = [] # contours that pass the filter

    for contour in contours:
        aspect_ratio, length_in_microns, extent = get_min_box(contour, pixel_to_micrometer)

        if aspect_ratio > 2.5 and PARTICLE_MIN_LENGTH_MICROMETERS <= length_in_microns and extent >0.3: #and PARTICLE_MIN_LENGTH_MICROMETERS <= length_in_microns and extent > 0.5: #and PARTICLE_MIN_LENGTH_MICROMETERS <= length_in_microns <= PARTICLE_MAX_LENGTH_MICROMETERS:
            filtered_contours.append(contour)
    
    num_extended = len(filtered_contours)

    return num_extended, filtered_contours



#--------------CONTRACTED R BODIES-------------------------------

def get_thresh(image): # image must be binary 

    # std dev based thresholding 
    mean, stddev = cv2.meanStdDev(image)
    thresh_value = int(mean - 1.5*stddev) # threshold by 1.5 standard deviations below the mean (closer to black, as the coiled r bodies are). Using this rather than a predefined threshold value because the brightness changes across images. 

    ret1, thresh = cv2.threshold(image,thresh_value,255, type = cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # cv2.imwrite("thresholded.tif", opened)
    
    return opened

#Watershedding to try to separate clumps. Doesn't work to well, I think because of how small the R-bodies are in these images. 
def get_watershed(image, colour): # colour is the real microscope image that is inputted, image is thresholded 
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    sure_bg = cv2.dilate(image, kernel, iterations = 1) # play around with the number of iterations. 
    # cv2.imwrite("sure_bg.tif", sure_bg)

    #Distance Transform - calculates the distance of each white pixel to the closest black pixel. Stored in the dist variable. 
    dist = cv2.distanceTransform(image, cv2.DIST_L2, 5)

    # foreground area
    ret, sure_fg = cv2.threshold(dist, 0.3*dist.max(), 255, cv2.THRESH_BINARY) # Play around with the scale factor
    sure_fg = sure_fg.astype(np.uint8)
    # cv2.imwrite("sure_fg.tif", sure_fg)

    #unknown area 
    unknown = cv2.subtract(sure_bg, sure_fg)
    # cv2.imwrite("unknown.tif", unknown)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers += 1
    markers[unknown == 255] = 0 # serves as the input into the watershed function. 

    markers = cv2.watershed(colour, markers) # requires the unaltered microscope image, not in gray scale?
    labels = np.unique(markers)

    coils = [] 
    for label in labels[2:]:  
    
    # Create a binary image in which only the area of the label is in the foreground 
    #and the rest of the image is in the background   
        target = np.where(markers == label, 255, 0).astype(np.uint8)
    
    # Perform contour extraction on the created binary image
        contours, hierarchy = cv2.findContours(
            target, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        coils.append(contours[0])


    
    return coils

# Function for handling contracted r bodies. Return the number of contours and the contours
def get_coiled(file):
    # Load image
    image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)  # Load .tif image
    if image is None:
        raise ValueError("Failed to load image. Check the file path and format.")
    
    watershed_image = cv2.imread(file)
    thresh = get_thresh(image)
    w_contours = get_watershed(thresh, watershed_image)

    num_coiled = len(w_contours)

    return num_coiled, w_contours

#function to draw the extended and coiled contours on each image. Returns an image that can be saved. 
def draw_contours(image, coiled_cnt, extended_cnt):
    output = cv2.imread(image)  # Convert to BGR for visualization
    if len(coiled_cnt)>0:
        for contour in coiled_cnt:
            # Draw the contour in green
            cv2.drawContours(output, [contour], -1, (0, 255, 0), 1)  #green contours

    if len(extended_cnt)>0:
        for contour in extended_cnt:
            # Draw the contour in red
            cv2.drawContours(output, [contour], -1, (255, 0, 0), 1)  # Red contours
        
    return output

folder = "C:/Users/carlo/Desktop/iGEM/Image Analysis/OutPut" # folder containing all the images to analyse
output_dir = "C:/Users/carlo/Desktop/iGEM/Image Analysis/output_contours/" # directory to store images with contours mapped on them

def processFolder(folder):

    directory = folder

    # ADD ARRAYS HERE THAT YOU WANT TO PASS MEASUREMENTS TO. APPEND THESE TO "data" after looping through a folder 
    files = [] # filenames for the file column
    contracted = [] # array to store number of contracted R bodies, for the csv
    extended = [] # as above, for extended R-bodies. 
    
    for file in os.listdir(directory):
        filename = os.path.join(directory, file)
        if file.endswith(".tif"): # CALL YOUR FUNCTIONS HERE TO MEASURE R-BODIES IN EACH IMAGE
            files.append(file)
            print(filename)
            num_coiled, coiled_contours = get_coiled(filename)
            contracted.append(num_coiled)

            num_extended, extended_contours = get_extended(filename)
            extended.append(num_extended)

            image_contours = draw_contours(filename, coiled_contours, extended_contours)
            cv2.imwrite("{}contours_{}".format(output_dir, file), image_contours)



    data = {
        "File": files, 
        "Contracted": contracted,
        "Extended": extended
    }

    spreadsheet = pd.DataFrame(data)
    
    spreadsheet.to_csv("rbody_quantification.csv")
     

processFolder(folder)

