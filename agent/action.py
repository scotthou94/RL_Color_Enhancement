# action
# @Author       Zihan
# @Mod Date     11/25/2018
# @Description  12 actions possibly performed by agent after deep learning

import os, sys
import colorsys
import numpy as np
from PIL import Image, ImageEnhance
pj_path = os.path.dirname(os.path.realpath(__file__))
pj_path = pj_path.split('/')
pj_path = '/'.join(pj_path[:-1])
if pj_path not in sys.path:
    sys.path.insert(0, pj_path)
import feature_extractor.color_histogram.color_histogram as color_histogram
if sys.platform == 'darwin':
    import matplotlib
    matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

# f is the increase/decrease factor, positive f -> increase, negative f -> decrease
# manipulate contrast
def mani_c(img_path, f):   # input image, direction
    img = Image.open(img_path)
    enhancer = ImageEnhance.Contrast(img)
    new_img = enhancer.enhance(f)
    return np.array(new_img)

# manipulate saturation
def mani_s(img_path, f):   # input image, direction
    rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)
    img = Image.open(img_path)
    arr = np.array(np.asarray(img), dtype=float)
    r, g, b = np.rollaxis(arr, axis = -1)
    h, s, v = rgb_to_hsv(r, g, b)
    s *= f
    s /= np.max(s)
    r, g, b = hsv_to_rgb(h, s, v)
    new_img = np.dstack((r,g,b))
    return new_img.astype('uint8')

# manipulate brightness
def mani_b(img_path, f):   # input image, direction
    img = Image.open(img_path)
    enhancer = ImageEnhance.Brightness(img)
    new_img = enhancer.enhance(f)
    return np.array(new_img)

# manipulate white balance
# pair should be 'rg', 'gb' or 'rb'
def mani_wb(img_path, pair, f):   # input image, direction, pair (red and green, green and blue or red and blue)
    img = np.array(Image.open(img_path), dtype=float)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if pair == 'rg':
        r *= f
        g *= f
    elif pair == 'gb':
        g *= f
        b *= f
    elif pair == 'rb':
        r *= f
        b *= f
    else:
        raise ValueError('Invalid pair')
    new_img = np.dstack((r, g, b))
    return new_img.astype('uint8')

''' # test client
if __name__ == "__main__":
    img_path = sys.argv[1]
    img = Image.open(img_path)
    img.show()
    img = np.array(img)
    #new_img = mani_s(img_path, 1.5)
    new_img = mani_wb(img_path, 'rb', 1.5)
    ni = Image.fromarray(new_img, 'RGB')
    ni.show()
    
    hg_img = color_histogram.get_histogram(img)
    hg_new_img = color_histogram.get_histogram(new_img)
    color = ['r', 'g', 'b'] # L = red, a = green, b = blue

    plt.figure()
    for i, c in enumerate(color):
        plt.plot(hg_img[i], color = c)
        plt.xlim([0,32])
    
    plt.figure()
    for i, c in enumerate(color):
        plt.plot(hg_new_img[i], color = c)
        plt.xlim([0,32])

    plt.show()
'''