# lbp.py
# Name: Dushyant Panchal
# Rollno: 2018033

import sys
import cv2
import numpy as np

# importing the image in grayscale mode
img=cv2.imread(sys.argv[1], cv2.IMREAD_GRAYSCALE)

# image dimensions
size = img.shape

# splits per dimension
splits_per_dim = 2

# patch sizes
patch_size = (size[0]//splits_per_dim, size[1]//splits_per_dim)

# stride size
stride = (1,1)


# feature vector
features = []

# iterate over the patches
for a in range(splits_per_dim):
	for b in range(splits_per_dim):
		# extract a patch
		patch = img[patch_size[0]*a:patch_size[0]*(a+1), patch_size[1]*b:patch_size[1]*(b+1)]
		
		# histogram init
		histo = np.zeros(256)

		# add padding for easy computation
		patch = np.pad(patch, (1,1), 'constant', constant_values=(0,0))
		
		# iterating over 3x3 sub matrices of the patch under consideration
		for i in range(1, patch.shape[0]-1, stride[0]):
			for j in range(1, patch.shape[1]-1, stride[1]):

				# calculating the binary vector and its decimal equivalent
				val = 0
				center = patch[i,j]
				
				val*=2
				val+= int(patch[i-1, j-1]>=center)

				val*=2
				val+= int(patch[i-1, j]>=center)

				val*=2
				val+= int(patch[i-1, j+1]>=center)

				val*=2
				val+= int(patch[i, j+1]>=center)

				val*=2
				val+= int(patch[i+1, j+1]>=center)

				val*=2
				val+= int(patch[i+1, j]>=center)

				val*=2
				val+= int(patch[i+1, j-1]>=center)

				val*=2
				val+= int(patch[i, j-1]>=center)

				histo[val] += 1

		# collecting the features
		features.extend(histo)


# Preview the results
print(len(features),'features collected.')
print(np.array(features))

