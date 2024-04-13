from typing import final
import cv2
import numpy as np
from numpy import hsplit, who
from matplotlib import pyplot as plt
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

    print(f"{image_input} variance: {variance}")
    return variance


def noise_reduction(image_in, noise):
    noise = noise_detection(image_in)
    if noise > 10000:
        return cv2.fastNlMeansDenoisingColored(image_in, None, 30, 10, 7, 21)
    elif (noise <= 10000): #& (noise > 3000):
        return cv2.fastNlMeansDenoisingColored(image_in, None, 5, 5, 7, 21)

    else:
        return image_in

#Good sharpe 
def sharpen(img_in, kernal_size):

    if img_in is None:
        print("No image ERROR")
        return None

    blurred =  cv2.GaussianBlur(img_in, kernal_size, 0)
    
    cv2.imwrite('changed_img.jpg',img_in)
    unsharp_mask = cv2.addWeighted(img_in, 1.5, blurred, -0.5, 0)
    cv2.imwrite('changed_img.jpg', unsharp_mask)
    
    return unsharp_mask

    #plt.figure(figsize=(10,5))
    #plt.subplot(121), plt.imshow(cv2.cvtColor(img_touse, cv2.COLOR_BGR2RGB)), plt.title('Oringinal Image')
    #plt.subplot(122), plt.imshow(cv2.cvtColor(unsharp_mask, cv2.COLOR_BGR2RGB)), plt.title('Sharpend')

def clean_image(image_in):
    noise_level = noise_detection(image_in)

    image = cv2.imread(image_in)

    noise_reduced_image = noise_reduction(image, noise_level)

    if noise_level <= 3000:
        final_image = sharpen(noise_reduced_image, (1, 1))

    else:
        final_image = noise_reduced_image

    noise_detection(final_image)

    return final_image

#clean_image('example_noisy_image.png')

