# Q3.py
# Name: Dushyant Panchal
# Rollno: 2018033


import cv2
import numpy as np
import time



start = time.time()
# importing the image
img = cv2.imread('iiitd2.png')

# generate histogram
colors,hist=np.unique(img.reshape(-1,3), return_counts=True, axis=0)
colors = colors.astype('int32') # convert to avoid data loss due to uint8


print('histogram generated')


# number of unique colors    
n_cols = colors.shape[0]

# saliency map
saliency = np.zeros(n_cols)

# generate saliency map for unique colors
for col_i in range(n_cols):    
    # take a color
    color_i = colors[col_i]
    
    # calculate its chebyshev distance from all unique colors
    cheby = np.max(np.abs(colors - color_i), axis=1)/256

    # find the saliency for this color
    saliency[col_i] = np.dot(hist,cheby)

# normalizing
saliency/=np.max(saliency)


print('saliency calc done')


# creating a color to saliency mapping
color_to_saliency = dict()
for col in range(n_cols):
    # generate a tuple for every unique color
    color_i = tuple(colors[col])

    # use this tuple as hash key to store corresponding saliency value
    color_to_saliency[color_i] = saliency[col]    


# generating saliency map for visualization
img_sal = np.zeros(img.shape[:2])
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        # generate tuple for every pixel's color
        c = tuple(img[i,j,:])
        
        # obtain the saliency value from the hashmap
        img_sal[i,j] = color_to_saliency[c]

end = time.time()
print(f"Runtime: {end-start}s")

cv2.imshow('saliency', img_sal)
cv2.waitKey(0)