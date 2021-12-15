# 2018033_CV_HW13.py
# Name: Dushyant Panchal
# Rollno: 2018033


import cv2 # only to read input image
import numpy as np

# input the images to np array
img1 = cv2.imread('Cap1.png')
img2 = cv2.imread('Cap2.png')

# some variables
num_imgs = 2
size = 400

# resize the images to a common dimension
img1 = cv2.resize(img1,(size,size))
img2 = cv2.resize(img2,(size,size))

# creating a feature space for kmeans
cluster_features = np.stack([img1,img2]).reshape(-1,3)
cluster_features = np.float32(cluster_features)

# running kmeans clustering
K = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
ret,label,center = cv2.kmeans(cluster_features,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# get cluster sizes and labels
_, nK = np.unique(label, return_counts=True)
label = label.reshape(2,size,size)
label1 = label[0,:,:]
label2 = label[1,:,:]

# generating corresponding cue 'q' measures
q = np.zeros((K,num_imgs))
for k in range(K):
    for j in range(num_imgs):
        label_j = label[j,:,:]
        q[k,j] = np.sum((label_j==k)*1)/nK[k]

# finding the variance
w_d = 1/(1+np.var(q, axis=1))
w_d /= np.max(w_d)
w_d = np.uint8(w_d)


# mapping the values back to the original images
map1 = np.zeros((size,size))
map2 = np.zeros((size,size))

for k in range(K):
    map1[label1==k] = w_d[k]
    map2[label2==k] = w_d[k]


# also mapping the clusters
cluster1 = np.zeros((size,size,3))
cluster2 = np.zeros((size,size,3))

for k in range(K):
    cluster1[label1==k] = center[k]/255
    cluster2[label2==k] = center[k]/255

# showing the results
cv2.imshow("img1", map1)
cv2.imshow("img2", map2)
cv2.imshow("cluster1", cluster1)
cv2.imshow("cluster2", cluster2)
cv2.waitKey(0)