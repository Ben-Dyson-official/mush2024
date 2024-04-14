""" A module designed to store utility functions for starmapper """
print("\nModule 'util' imported successfully.\n")

# import the necessary packages
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2
import os
import csv

from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

def classify(filename):
	"""Circles the stars of an input image, writes the output to a file"""
	image = cv2.imread(os.path.join('./flaskr/static/', filename))

	original_image = image # save the original image

	image = process_img(image)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 2)

	# threshold the image to reveal light regions in the
	# blurred image
	thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]

	# perform a series of erosions and dilations to remove
	# any small blobs of noise from the thresholded image
	thresh = cv2.erode(thresh, None, iterations=0)
	thresh = cv2.dilate(thresh, None, iterations=1)

	star_mask = get_star_mask(thresh)
	cluster_mask = get_cluster_mask(original_image)
	
	# find the contours in the mask, then sort them from left to
	# right
	cnts = cv2.findContours(star_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = contours.sort_contours(cnts)[0]
	star_num = 0	
	# loop over the contours
	for (i, c) in enumerate(cnts):
		# draw the bright spot on the image
		(x, y, w, h) = cv2.boundingRect(c)
		((cX, cY), radius) = cv2.minEnclosingCircle(c)
		cv2.circle(original_image, (int(cX), int(cY)), int(radius), (0, 255, 0), 3)
		#cv2.putText(image, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
		star_num += 1

	# trace the clusters
	cnts = cv2.findContours(cluster_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = contours.sort_contours(cnts)[0]
	cluster_num = 0
	for c in cnts:
		cv2.drawContours(original_image, [c], -1, (255, 255, 0), thickness=10)
		cluster_num += 1

	# show the output image
	cv2.imwrite(os.path.join('./flaskr/static/', filename), original_image)

	return star_num, cluster_num

# BEFORE CLASSIFY CLEANING

def process_img(image):
    return color_hist_change(clean_image(image))

def color_hist_change(colorimage):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    colorimage_b = clahe.apply(colorimage[:,:,0])
    colorimage_g = clahe.apply(colorimage[:,:,1])
    colorimage_r = clahe.apply(colorimage[:,:,2])
    
    colorimage_e = np.stack((colorimage_b, colorimage_g, colorimage_r), axis=2)
    colorimage_e.shape

    return colorimage_e

def clean_image(image_in):
    noise_level = noise_detection(image_in)

    noise_reduced_image = noise_reduction(image_in, noise_level)

    if noise_level <= 3000:
        final_image = sharpen(noise_reduced_image, (1, 1))

    else:
        final_image = noise_reduced_image

    noise_detection(final_image)

    return final_image

def noise_detection(image_input) -> float:  
    """Higer the variance the higher the noise"""
    
    """Higher the variance the higher the noise."""
    if isinstance(image_input, str):
        image = cv2.imread(image_input)
        if image is None:
            raise ValueError(f"Image at {image_input} could not be loaded.")
        grey_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif isinstance(image_input, np.ndarray):
        grey_scale = image_input
    else:
        raise TypeError("Input must be a file path string or a numpy array.")

    laplacian = cv2.Laplacian(grey_scale, cv2.CV_64F)
    variance = laplacian.var()

    #print(f"{image_input} variance: {variance}")
    return variance

def noise_reduction(image_in, noise):
    noise = noise_detection(image_in)
    if noise > 10000:
        return cv2.fastNlMeansDenoisingColored(image_in, None, 30, 10, 7, 21)
    elif (noise <= 10000):
        return cv2.fastNlMeansDenoisingColored(image_in, None, 5, 5, 7, 21)

    else:
        return image_in

def sharpen(img_in, kernal_size):
	
	if img_in is None:
		print("No image ERROR")
		return None

	blurred =  cv2.GaussianBlur(img_in, kernal_size, 0)

	unsharp_mask = cv2.addWeighted(img_in, 1.5, blurred, -0.5, 0)

	return unsharp_mask

def get_star_mask(thresh):
	# perform a connected component analysis on the thresholded
	# image, then initialize a mask to store only the "large"
	# components
	labels = measure.label(thresh, background=0)
	mask = np.zeros(thresh.shape, dtype="uint8")
	
	# loop over the unique components
	for label in np.unique(labels):
		# if this is the background label, ignore it
		if label == 0:
			continue
		# otherwise, construct the label mask and count the
		# number of pixels 
		labelMask = np.zeros(thresh.shape, dtype="uint8")
		labelMask[labels == label] = 255
		numPixels = cv2.countNonZero(labelMask)
		# if the number of pixels in the component is sufficiently
		# large, then add it to our mask of "large blobs"
		if numPixels > 1 and numPixels < 1000:
			mask = cv2.add(mask, labelMask)

	return mask

# CREATE MASK FOR BIG CLUSTERS
def get_cluster_mask(img):
	lower_val = np.array([0,0,0])
	upper_val = np.array([40,40,100])

	mask = cv2.inRange(img, lower_val, upper_val)

	contours, _ = cv2.findContours(mask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	selected_contours = []

	for contour in contours:
		area = cv2.contourArea(contour)
  
		if area > 1000: #this is the val we can change
			blank_image = np.zeros(mask.shape, np.uint8)
			selected_contours.append(contour)
			cv2.fillPoly(blank_image, pts=selected_contours, color=(255, 255, 255))

	out_img = cv2.bitwise_not(blank_image)
	
	return out_img

def check_model(image_path):
    model = load_model('model.h5')
    
    prepped_image = preprocess_image(image_path)

    prediction = model.predict(prepped_image)

    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class

def preprocess_image(path):
	# read in file
	img = cv2.imread(path)
	# mask the image
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 2) 
	img = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]
	# save masked image
	cv2.imwrite(path, img)

	img = load_img(path, target_size = (256, 256))
    
	a = img_to_array(img)
	a = np.expand_dims(a, axis = 0)
	a /= 255.
	return a

def read_csv(index):
    with open('constellation_facts.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        facts = list(reader)

        fact = facts[index]

    return fact
