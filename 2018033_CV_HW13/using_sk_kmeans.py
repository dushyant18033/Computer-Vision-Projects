# 2018033_CV_HW13.py
# Name: Dushyant Panchal
# Rollno: 2018033


import cv2
import numpy as np
from sklearn.cluster import KMeans

# input the images to np array
img1 = cv2.imread('Cap1.png')
img2 = cv2.imread('Cap2.png')
num_imgs = 2

size = 400

img1 = cv2.resize(img1,(size,size))
img2 = cv2.resize(img2,(size,size))


img = np.stack([img1,img2]).reshape(num_imgs*size,size,3)
cluster_features = np.float32(img.reshape(-1,3))

K=5
kmeans = KMeans(n_clusters=K).fit(cluster_features)

_, nK = np.unique(kmeans.labels_, return_counts=True)
labels = kmeans.labels_.reshape(num_imgs,size,size)


q = np.zeros((K,num_imgs))
for k in range(K):
    for j in range(num_imgs):
        label_j = labels[j,:,:]
        q[k,j] = np.sum((label_j==k)*1)/nK[k]

w_d = 1/(1+np.var(q, axis=1))
w_d /= np.max(w_d)
w_d = np.uint8(w_d)




maps = np.zeros(img.shape[:2])
labels = labels.reshape(maps.shape)
for k in range(K):
    maps[labels==k] = w_d[k]



clusters = np.zeros(img.shape)

for k in range(K):
    clusters[labels==k] = kmeans.cluster_centers_[k]/255

cv2.imshow("imgs", maps)
cv2.imshow("clusters", clusters)
cv2.waitKey(0)