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

kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
im = cv2.filter2D(img, -1, kernel)

sharp = np.hstack((img, im))
cv2.imwrite('sharp.png', sharp)


blurred =  cv2.GaussianBlur(img, (9,9), 0)

unsharp_mask = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)

plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(img, cmap='grey'), plt.title('Oringinal Image')
plt.subplot(122), plt.imshow(unsharp_mask, cmap='grey'), plt.title('Sharpend')

# Laplce 

laplacian = cv2.Laplacian(img, cv2.CV_64F)
laplacian = cv2.convertScaleAbs(laplacian)

sharpened = cv2.addWeighted(img, 1.5, laplacian, -0.5, 0)

plt.figure(figsize=(15, 7))
plt.subplot(131), plt.imshow(img, cmap='grey'), plt.title("Original")
plt.subplot(132), plt.imshow(img, cmap='grey'), plt.title("Sharpend2")
plt.show()


