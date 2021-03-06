# color histogram
# @Author       Zihan
# @Mod Date     11/13/2018
# @Description  Input color image in RGB space, convert it into CIELab space, apply CLAHE to lightness channel, and calculate histogram values by OpenCV.

import sys
import cv2  # OpenCV library
import numpy as np
# need this code below to avoid some strange error of matplotlib on Mac
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

def get_histogram(img):
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)  # convert color space from RGB to Lab
    l_channel, a_channel, b_channel = cv2.split(lab_img)    # separate L, a and b channels
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8, 8))
    clahe_l_channel = clahe.apply(l_channel)    # apply CLAHE to lightness channel
    equalized_img = cv2.merge((clahe_l_channel, a_channel, b_channel))
    histogram = []   # L, a, b in bins for histogram
    for i in range(3):
        histogram.append(cv2.calcHist([equalized_img], [i], None, [32], [0,256]))   # 32 bins
    return _normalize(histogram)

# use formula: normalized_x = (x - min) / (max - min)
def _normalize(histogram):
    normalized_histogram = np.array([np.zeros((32, 1)), np.zeros((32, 1)), np.zeros((32, 1))])
    for i in range(3):
        maxv = np.max(histogram[i])
        minv = np.min(histogram[i])
        for j in range(32):
            normalized_histogram[i][j] = (histogram[i][j] - minv) / (maxv - minv)
    return normalized_histogram

''' # test client
if __name__ == '__main__':
    img_path = sys.argv[1]
    img = cv2.imread(img_path, 1) # load a color image in np.array
    #img = np.random.randint(256, size=(200,200,3), dtype='uint8')
    hg = get_histogram(img)
    print(type(hg))
    color = ['r', 'g', 'b'] # L = red, a = green, b = blue
    for i, c in enumerate(color):
        plt.plot(hg[i], color = c)
        plt.xlim([0,32])
        print(c)
        print(hg[i])
    plt.show()
'''