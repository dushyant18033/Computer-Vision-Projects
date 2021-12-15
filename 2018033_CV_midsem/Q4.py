# Q4.py
# Name: Dushyant Panchal
# Rollno: 2018033

import cv2
import numpy as np

# importing the image in grayscale mode
img = cv2.imread('iiitd1.png', cv2.IMREAD_GRAYSCALE)
img_lbp = np.zeros(img.shape)
img = np.pad(img, (1,1), 'constant', constant_values=(0,0))


# iterating over 3x3 sub matrices of the image
for i in range(1, img.shape[0]-1):
	for j in range(1, img.shape[1]-1):

		# calculating the binary vector and its decimal equivalent
		val = 0
		a = int(img[i,j])
		list_b = [
			img[i-1, j-1],
			img[i-1, j],
			img[i-1, j+1],
			img[i, j+1],
			img[i+1, j+1],
			img[i+1, j],
			img[i+1, j-1],
			img[i, j-1]
		]

		# iterate over the neighbors clock-wise
		for b in list_b:
			b = int(b)
			val*=2
			val += int( (min(a,b)/(0.00001 + max(a,b)) )>=0.5 )
		img_lbp[i-1,j-1] = val

# show the resulting feature map as an image
cv2.imshow('feature map', img_lbp.astype('uint8'))
cv2.waitKey(0)