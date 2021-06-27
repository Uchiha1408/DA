import cv2
import numpy as np
img = cv2.imread('E:\\6.jpg')
sift = cv2.SIFT_create()
kp = sift.detect(img,None)
img=cv2.drawKeypoints(img, kp,None)
cv2.imshow('img',img)
