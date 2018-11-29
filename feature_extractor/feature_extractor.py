import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16

class ContextExtractor:
    def __init__(self):
        # Use the first six layers of vgg16
        convbase = VGG16(weights='imagenet',
                         include_top=True)
        layers = convbase.layers[:21]
        self.model = tf.keras.models.Sequential()

        for layer in layers:
            self.model.add(layer)

    def __call__(self, data):
        # data is a RGB numpy array represents an image
        # Should have size (224,224,3)
        # Return the feature vector of shape (4096,)
        assert data.shape == (224,224,3)
        data = np.array([data])
        return self.model.predict(data)[0]

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