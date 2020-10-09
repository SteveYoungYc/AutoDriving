import cv2
import numpy as np


def roi_mask(img, vertices):  # vertices 代表4个点坐标，坐标（横坐标，纵坐标）
    mask = np.zeros_like(img)
    mask_color = 255  # 代表白色
    cv2.fillPoly(mask, vertices, mask_color)
    # cv2.imshow(\"A1\",mask)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


img = cv2.imread('D:\\SJTU\\Auto Driving\\Lane.jpg')
imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
cv2.imshow('image', imgGray)

imgGua = cv2.GaussianBlur(imgGray, (3, 3), 0)
cv2.imshow('Gaussian', imgGua)

imgCanny = cv2.Canny(imgGua, 100, 200, 3)
cv2.imshow('Canny', imgCanny)

roi_vtx = np.array([[(0, img.shape[0]), (460, 325), (520, 325), (img.shape[1], img.shape[0])]])
roi_edges = roi_mask(edges, roi_vtx)

cv2.waitKey(0)

cv2.destroyAllWindows()
