# Name: Dushyant Panchal
# Rollno: 2018033

import sys
import cv2
import numpy as np

# importing the image
image = cv2.imread(sys.argv[1])
img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# the filter kernels
kernel = np.array([[-1,-1,-1],[0,0,0],[1,1,1]], dtype=np.float32)
kernel_x = kernel
kernel_y = kernel.T


# edge maps
Ex = np.zeros(img.shape)
Ey = np.zeros(img.shape)

for i in range(1,img.shape[0]-1):
    for j in range(1,img.shape[1]-1):

        neighbors = [
			[ img[i-1, j-1], img[i-1, j], img[i-1, j+1] ],
			[ img[i, j-1], img[i, j], img[i, j+1], ],
            [ img[i+1, j-1], img[i+1, j], img[i+1, j+1] ],
        ]
        neighbors = np.float32(neighbors)

        # applying the filter to 3x3 window
        for a in range(3):
            for b in range(3):
                # horizontal convolution
                Ex[i,j] += neighbors[a,b]*kernel_x[a,b]
                # vertical convolution
                Ey[i,j] += neighbors[a,b]*kernel_y[a,b]


Ex /= np.max(Ex)
Ey /= np.max(Ey)

E = np.sqrt(Ex**2 + Ey**2)
E /= np.max(E)

cv2.imshow("input image", image)
cv2.imshow("Ex", Ex)
cv2.imshow("Ey", Ey)
cv2.imshow("E", E)
cv2.waitKey(0)