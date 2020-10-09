import cv2
import numpy as np

img = cv2.imread('D:\\SJTU\\Auto Driving\\Lane.jpg')
cv2.imshow('image', img)

sp = img.shape
height = sp[0]
width = sp[1]

new = np.zeros((height, width, 3), np.uint8)
for i in range(height):
    for j in range(width):
        img[i][j] = 0.11 * img[i][j][0] + 0.59 * img[i][j][1] + 0.3 * img[i][j][2]

cv2.imshow('gray', img)
cv2.waitKey(0)

cv2.destroyAllWindows()
