# 2018033_CV_HW1.py
# Name: Dushyant Panchal
# Rollno: 2018033


import cv2 # only to read input image
import numpy as np
from collections import deque

# input the image to np array
image = cv2.imread('Project1.png', cv2.IMREAD_GRAYSCALE)


# threshold to convert to binary image
image[image<128] = 0
image[image>=128] = 1

# accumulator to count objects
count = 0

# dimensions of the image
m,n = image.shape


# iterate through the pixels
for a in range(m):
	for b in range(n):

		# if unvisited white pixel found
		if image[a,b]==1:

			# increment the counter to denote new object
			count+=1

			# find all neighboring white pixels and mark
			# them as visited by setting them to black
			queue = deque(((a,b),))
			image[a,b]=0

			# similar to BFS technique
			while len(queue)>0:
				i,j = queue.popleft()

				if i>0 and image[i-1,j]==1:
					image[i-1,j]=0
					queue.append((i-1,j))

				if j>0 and image[i,j-1]==1:
					image[i,j-1]=0
					queue.append((i,j-1))
				
				if i<m-1 and image[i+1,j]==1:
					image[i+1,j]=0
					queue.append((i+1,j))
				
				if j<n-1 and image[i,j+1]==1:
					image[i,j+1]=0
					queue.append((i,j+1))

# print the final count
print(count)
