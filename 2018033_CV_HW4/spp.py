# spp.py
# Name: Dushyant Panchal
# Rollno: 2018033

import sys
import cv2
import numpy as np

# importing the image.
img=cv2.imread(sys.argv[1])

# resize the image to 240x240 as suggested.
img = cv2.resize(img, (240,240))

# feature vector
features=[]

# perform the splits
for split in [2,3,4,5]:
	
	# split segment dimensions
	seg_size = 240//split

	# iterate through the splits
	for i in range(split):
		for j in range(split):
			
			# iterate over the channels
			for col in range(3):
				
				# extract a channel from the split
				part = img[seg_size*i:seg_size*(i+1), seg_size*j:seg_size*(j+1), col]
				
				# append mean and std to the feature vector
				features.append(np.mean(part))
				features.append(np.std(part))


# Preview the results
print(len(features),'features collected.')
print(np.array(features))