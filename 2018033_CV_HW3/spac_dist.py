# spac_dist.py
# Name: Dushyant Panchal
# Rollno: 2018033

import sys
import cv2
import numpy as np

# importing the image
img=cv2.imread(sys.argv[1])

# saliency accumulator
saliency = np.zeros((img.shape[0],img.shape[1]))

# calc diagonal length
diagonal = (img.shape[0]**2 + img.shape[1]**2)**0.5

# for normalizing color component
col_diff_max = ((255**2)*3)**0.5


# Calculating Saliency
for i in range(img.shape[0]):
	print(i*100/img.shape[0],"%", end='\r')

	for j in range(img.shape[1]):
		col_ij = img[i,j,:].astype('int')

		for a in range(img.shape[0]):
			for b in range(img.shape[1]):

				# color dist component
				col_ab = img[a,b,:].astype('int')
				col_dist = np.sum( (col_ij - col_ab)**2 )**0.5
				
				# spacial dist component
				sp_dist = ( (i-a)**2 + (j-b)**2 )**0.5

				# combining
				saliency[i,j] += (col_dist/col_diff_max)*np.exp(-(sp_dist/diagonal))

# normalizing
saliency/=np.max(saliency)


# displaying the results
cv2.imshow('imput', img)
cv2.imshow('saliency', (saliency))
cv2.waitKey(0)