# interactive_saliency.py
# Name: Dushyant Panchal
# Rollno: 2018033

import sys
import cv2
import numpy as np

# importing the image in grayscale mode
img=cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

# image dimensions
size = img.shape

# selected patches
fg_patch = img[140:140+75, 140:140+50]  # 140,140 (75x50)
bg_patch = img[225:225+75, 285:285+50]  # 225,285 (75x50)

patch_shape = fg_patch.shape

# histograms
fg_hist = np.zeros(256)
bg_hist = np.zeros(256)

for i in range(patch_shape[0]):
	for j in range(patch_shape[1]):
		fg_hist[fg_patch[i,j]]+=1
		bg_hist[bg_patch[i,j]]+=1

# normalizing the distribution
fg_hist/=np.max(fg_hist)
bg_hist/=np.max(bg_hist)


# fg map
img_fg = np.zeros(size)
for i in range(size[0]):
	for j in range(size[1]):
		col = img[i,j]
		img_fg[i,j] = fg_hist[col]


# bg map
img_bg = np.zeros(size)
for i in range(size[0]):
	for j in range(size[1]):
		col = img[i,j]
		img_bg[i,j] = bg_hist[col]


# saliency map
img_sal = (img_fg + (1-img_bg))/2


# displaying the results
cv2.imshow('fg map',img_fg)
cv2.imshow('bg map',img_bg)
cv2.imshow('saliency map',img_sal)
cv2.waitKey()