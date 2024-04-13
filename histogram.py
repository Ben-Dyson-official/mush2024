import cv2
from matplotlib import pyplot as plt
import numpy as np

img = cv2.imread('test_image_grey.jpg', 0)

clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8, 8))
cl1 = clahe.apply(img)

equ = cv2.equalizeHist(img)
res =  np.hstack((img, equ))
cl1 = np.hstack((img, cl1))

cv2.imwrite('clahe.png', cl1)
cv2.imwrite('res.png', res)
