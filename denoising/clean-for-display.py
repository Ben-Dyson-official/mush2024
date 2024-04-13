from typing import final
import cv2
from cv2.typing import MatLike
import numpy as np
from numpy import hsplit, who
from matplotlib import pyplot as plt 
import noiseDetection as nd
from PIL import Image 
#import pyheif

def read_heic(file_path):
    # Read the HEIC file
    heic_file = pyheif.read(file_path)
    
    # Convert to PIL Image
    image = Image.frombytes(
        heic_file.mode, 
        heic_file.size,
        heic_file.data,
        "raw",
        heic_file.mode,
        heic_file.stride,
    )
    
    # Convert to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
    return image

def clean_for_displau(image):
    image = cv2.imread(image)

    if image is None:
        raise ValueError("Image could not be loaded")

    nd.noise_detection(image)

    image = nd.clean_image(image)
    
    return image

image = clean_for_displau('Images/bear.png')

cv2.imwrite('Out.jpg', image)

