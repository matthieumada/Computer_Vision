import cv2
# to understand the pyramid 

img = cv2.imread("/home/delinm/Documents/Computer_Vision/LAB_8/files/1017.png")
img2 = cv2.pyrDown(img)       # half the size
img4 = cv2.pyrDown(img2)      #half of the half of the size 

cv2.imshow("1x", img)
cv2.imshow("1/2x", img2)
cv2.imshow("1/4x", img4)
cv2.waitKey(0)
