import cv2

img = cv2.imread('D:\\CV\\Python\\Battery.JPG', 0)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

epsilon = 0.1 * cv2.arcLength(img_gray, True)
approx = cv2.approxPolyDP(img_gray, epsilon, True)

while 1:
    cv2.imshow('gray', img_gray)
    #cv2.imshow('approx', approx)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
cv2.destroyAllWindows()
