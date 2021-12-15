# 2018033_CV_HW2.py
# Name: Dushyant Panchal
# Rollno: 2018033


import sys
import cv2
import numpy as np

# importing the image
img=cv2.imread(sys.argv[1])

# creating a grayscale copy
img_gr = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# calculating the otsu threshold
min_v_weighted = float("inf")
otsu_t = -1

for t in range(0,256):
	# set I0
	ind0 = img_gr<t
	I0 = img_gr[ind0]
	v0 = np.var(I0)

	# set I1
	ind1 = img_gr>=t
	I1 = img_gr[ind1]
	v1 = np.var(I1)

	# weighted variance
	v_weighted = v0*I0.size + v1*I1.size

	# find 't' for min weighted variance
	if v_weighted < min_v_weighted:
		min_v_weighted = v_weighted
		otsu_t = t

# found otsu threshold
print("Otsu threshold:",otsu_t)



"""selecting the background and foreground"""
bg=""



# boundary pixels composition
top = img_gr[0, :-1]
left = img_gr[1:, 0]
right = img_gr[:-1, -1]
bottom = img_gr[-1, 1:]
I0_border = np.mean(top<otsu_t) + np.mean(left<otsu_t) + np.mean(right<otsu_t) + np.mean(bottom<otsu_t)
I0_border/=4

# only assumption 2
# if I0_border>0.5:
# 	bg="I0"
# else:
# 	bg="I1"


# centre pixels composition
m,n = img_gr.shape
x,y = m//5, n//5
img_centre = img_gr[2*x:3*x, 2*y:3*y]
I0_centre = np.mean(img_centre<otsu_t)

# only assumption 1
# if I0_centre<=0.5:
# 	bg="I0"
# else:
# 	bg="I1"



# combining both assumptions 1 & 2
if (I0_border/I0_centre)>=1:
	bg="I0"
else:
	bg="I1"



# showing the results
if bg=="I0":	# if I0 is the background
	img0 = img.copy()
	ind0 = img_gr<otsu_t
	img0[:,:,0][ind0] = 255
	img0[:,:,1][ind0] = 0
	img0[:,:,2][ind0] = 0

	cv2.imshow('I0-bg, I1-fg',img0)
	cv2.waitKey(0)

if bg=="I1":	# if I1 is the background
	img1 = img.copy()
	ind1 = img_gr>=otsu_t
	img1[:,:,0][ind1] = 255
	img1[:,:,1][ind1] = 0
	img1[:,:,2][ind1] = 0

	cv2.imshow('I0-fg, I1-bg',img1)
	cv2.waitKey(0)


