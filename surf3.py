import cv2
import matplotlib.pyplot as plt
import math
import numpy as np
img1 = cv2.imread('E:\\10.png')
img2 = cv2.imread('E:\\11.png')
surf = cv2.SURF_create()
kp1, des1= surf.detectAndCompute(img1,None)
kp2, des2 = surf.detectAndCompute(img2,None)
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key= lambda match : match.distance)
matched_imge = cv2.drawMatches(img1, kp1, img2, kp2, matches[:30], None)
cv2.imshow("Matching Images", matched_imge)
