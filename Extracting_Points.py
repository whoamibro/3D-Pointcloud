import numpy as np
import cv2
from matplotlib import pyplot as plt
import os

M_IMGDIR = './asset/matching/'
C_IMGDIR = './asset/color/'

img_list_for_matching = os.listdir(M_IMGDIR)

img_data = []

for i in img_list_for_matching:
    img_data.append(cv2.imread(M_IMGDIR+i, 0))

print(img_data)

sift = cv2.xfeatures2d.SIFT_create()

img_kd_data = []

for i in range(2):
    key_desc_Data = {}
    key_desc_Data["kp"],  key_desc_Data['desc'] = sift.detectAndCompute(img_data[i], None)
    img_kd_data.append(key_desc_Data)
    print(key_desc_Data)

bruteforceMatcher = cv2.BFMatcher()
matches = bruteforceMatcher.knnMatch(img_kd_data[0]['desc'], img_kd_data[1]['desc'], k=2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

#newimg = cv2.drawMatchesKnn(img1 = img_data[0], keypoints1 = img_kd_data[0]['kp'], img2 = img_data[1], keypoints2 = img_kd_data[1]['kp'], outImg = None, matches1to2 = good, flags=2)
newimg = cv2.drawKeypoints(image = img_data[1], keypoints = img_kd_data[1]['kp'], outImage=None, color = (0,255,0), flags=2)

plt.imshow(newimg)

plt.show()