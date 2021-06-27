import cv2
import numpy as np
img = cv2.imread('E:\\build.png')
surf = cv2.SURF_create()
kp = surf.detect(img,None)
img=cv2.drawKeypoints(img, kp,None)
cv2.imshow('img',img)
