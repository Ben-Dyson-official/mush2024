import numpy as np
import cv2
from matplotlib import pyplot as plt

def denoise(img):
	# denoising of image saving it into dst image 
	dst = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 15) 
	return dst

# Reading image from folder where it is stored 
img = cv2.imread('saturn.png') 
dst = img

amount = int(input("How many denoises? "))

for i in range(amount):
	dst = denoise(dst);

# Plotting of source and destination image 
plt.subplot(121), plt.imshow(img) 
plt.subplot(122), plt.imshow(dst) 
  
plt.show()
