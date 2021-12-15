# Q1.py
# Name: Dushyant Panchal
# Rollno: 2018033


import cv2
import numpy as np

def calc_tss(img):
    """
    Returns the TSS of the pixel values given in 'img'.
    """
    if img.size == 0:   # handling empty set case
        return 0
    mu = np.mean(img)   # find the mean
    return np.sum((img-mu)**2)  # return TSS

def otsu_threshold(img_gr):
    """
    Returns the OTSU Threshold for the pixel values
    in 'img' considering the sum of TSS's.
    """
    # accumulator variables
    min_tss_sum = float("inf")
    otsu_t = -1

    # looping over all possible thresholds
    for t in range(0,256):
        
        # set I0
        ind0 = img_gr<t
        I0 = img_gr[ind0]
        tss0 = calc_tss(I0)

        # set I1
        ind1 = img_gr>=t
        I1 = img_gr[ind1]
        tss1 = calc_tss(I1)

        # unweighted tss sum
        tss_sum = tss0 + tss1

        # see if current 't' is better
        if tss_sum < min_tss_sum:
            min_tss_sum = tss_sum
            otsu_t = t
    print(min_tss_sum)
    # return the threshold calculated above
    return otsu_t

if __name__=="__main__":
    # importing the image
    img_gr = cv2.imread('iiitd1.png', cv2.IMREAD_GRAYSCALE)

    # find otsu threshold
    threshold = otsu_threshold(img_gr)
    print("Otsu threshold:",threshold)

    # image with set0 pixels set to white
    ind0 = img_gr<threshold
    img0 = np.zeros(img_gr.shape)
    img0[ind0]=255

    # image with set1 pixels set to white
    ind1 = img_gr>=threshold
    img1 = np.zeros(img_gr.shape)
    img1[ind1]=255

    # display visual results
    cv2.imshow(f'Set I0 (pixels<{threshold})', img0)
    cv2.imshow(f'Set I1 (pixels>={threshold})', img1)
    cv2.waitKey(0)