# color.py
# Name: Dushyant Panchal
# Rollno: 2018033

import sys
import cv2
import numpy as np

# importing the image
img=cv2.imread(sys.argv[1])

# accumulator for histogram
hist = np.zeros((256,256,256))

# calculating histogram
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		col = img[i,j,:]
		hist[col[0], col[1], col[2]] += 1

# max value from the histogram
h_max = np.max(hist)


# calculating saliency
saliency = np.zeros((img.shape[0],img.shape[1]))
for i in range(img.shape[0]):
	for j in range(img.shape[1]):
		col = img[i,j,:]
		h_i = hist[col[0], col[1], col[2]]
		saliency[i,j] = (h_max-h_i)/h_max


# displaying the results
cv2.imshow('imput', img)
cv2.imshow('saliency', (saliency))
cv2.waitKey(0)