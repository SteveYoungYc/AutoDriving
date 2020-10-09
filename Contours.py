import cv2

img = cv2.imread('D:\\CV\\Python\\Battery.JPG', 0)
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
# ret, thresh = cv2.threshold(img_gray, 127, 255, 0)
# img1 = cv2.convertMaps()

while 1:
    cv2.imshow('gray', img_gray)
    cv2.imshow('image', img)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
cv2.destroyAllWindows()
