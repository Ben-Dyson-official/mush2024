import cv2
import numpy as np

# read in image
img0 = cv2.imread("star_hubble.jpg")

# convert to greyscale
img1 = cv2.cvtColor(img0, cv2.COLOR_BGR2GRAY)

# apply Gaussian blur to reduce noise
img2 = cv2.GaussianBlur(img1, (7, 7), 0)

# apply thresholding to create binary image
#img3 = cv2.threshold(img2)
ret, img3 = cv2.threshold(img1,200,255,cv2.THRESH_BINARY)

# apply the hough circle transform to detect circles
circles = cv2.HoughCircles(img3, cv2.HOUGH_GRADIENT, 10, 5, maxRadius=50)
print(circles)

# draw the circles
if circles is not None:
	circles = np.round(circles[0, :]).astype("int")
	print(circles)
	for (x, y, r) in circles:
		cv2.circle(img0, (x, y), r, (0, 255, 0), 4)

# display the image
cv2.imshow("stars", img3)
cv2.waitKey()
cv2.imshow("stars", img0)

# write the file
#cv2.imwrite("result.png", img3)

# wait to be closed
cv2.waitKey()
