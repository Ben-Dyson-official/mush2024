import os
import cv2


for dir1 in os.listdir("C:/Users/James/Desktop/mush2024/constellation_data"):
    for dir2 in os.listdir("C:/Users/James/Desktop/mush2024/constellation_data/" + dir1):
        for filename in os.listdir("C:/Users/James/Desktop/mush2024/constellation_data/" + dir1 + "/" + dir2):
            img = cv2.imread("C:/Users/James/Desktop/mush2024/constellation_data/" + dir1 + "/" + dir2 + "/" + filename)
            
            print(filename)
            
            # mask the image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 2) 
            img = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)[1]
            
            print(img)
            
            cv2.imwrite("C:/Users/James/Desktop/mush2024/constellation_data_bw/" + dir1 + "/" + dir2 + "/" + filename, img)