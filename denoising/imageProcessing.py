import cv2
import numpy as np 
from matplotlib import pyplot as plt 
import noiseDetection

colorimage = cv2.imread('starcolour-link.png')


def colorHistChange(colorimage):

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

    colorimage_b = clahe.apply(colorimage[:,:,0])
    colorimage_g = clahe.apply(colorimage[:,:,1])
    colorimage_r = clahe.apply(colorimage[:,:,2])
    
    colorimage_e = np.stack((colorimage_b, colorimage_g, colorimage_r), axis=2)
    colorimage_e.shape

    return colorimage_e

def process_Img(image):
    #cv2.imwrite('cleaned-image.jpg', colorHistChange(noiseDetection.clean_image(image)))
    return colorHistChange(noiseDetection.clean_image(image))

#process_Img('star_hubble.jpg')
