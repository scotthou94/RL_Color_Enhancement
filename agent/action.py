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

# af is the positive actual factor, e.g. af = 0.05, then action will increase by 1 + af = 1.05 and decrease by 1 - af = 0.95
# actionlst[0]: increase contrast, actionlst[1]: decrease contrast
# actionlst[2]: increase saturation, actionlst[3]: decrease saturation
# actionlst[4]: increase brightness, actionlst[5]: decrease brightness
# actionlst[6]: increase red and green, actionlst[7]: decrease red and green
# actionlst[8]: increase green and blue, actionlst[9]: decrease green and blue
# actionlst[10]: increase red and blue, actionlst[11]: decrease red and blue
def actionlst():
    action_lst = [_mani_c_inc, _mani_c_dec, _mani_s_inc, _mani_s_dec, _mani_b_inc, _mani_b_dec, _mani_wb_rg_inc, _mani_wb_rg_dec, _mani_wb_gb_inc, _mani_wb_gb_dec, _mani_wb_rb_inc, _mani_wb_rb_dec]
    return action_lst

def _mani_c_inc(img_arr, af):
    return mani_c(img_arr, 1+af)

def _mani_c_dec(img_arr, af):
    return mani_c(img_arr, 1-af)

def _mani_s_inc(img_arr, af):
    return mani_s(img_arr, 1+af)

def _mani_s_dec(img_arr, af):
    return mani_s(img_arr, 1-af)

def _mani_b_inc(img_arr, af):
    return mani_b(img_arr, 1+af)

def _mani_b_dec(img_arr, af):
    return mani_b(img_arr, 1-af)

def _mani_wb_rg_inc(img_arr, af):
    return mani_wb(img_arr, 'rg', 1+af)

def _mani_wb_rg_dec(img_arr, af):
    return mani_wb(img_arr, 'rg', 1-af)

def _mani_wb_gb_inc(img_arr, af):
    return mani_wb(img_arr, 'gb', 1+af)

def _mani_wb_gb_dec(img_arr, af):
    return mani_wb(img_arr, 'gb', 1-af)

def _mani_wb_rb_inc(img_arr, af):
    return mani_wb(img_arr, 'rb', 1+af)

def _mani_wb_rb_dec(img_arr, af):
    return mani_wb(img_arr, 'rb', 1-af)

# f is the increase/decrease factor, f = 1 original image, f > 1 increase, 0 <= f < 1 decrease
# manipulate contrast
def mani_c(img_arr, f):   # input image, direction
    img = Image.fromarray(img_arr, 'RGB')
    enhancer = ImageEnhance.Contrast(img)
    new_img = enhancer.enhance(f)
    return np.array(new_img)

# manipulate saturation
def mani_s(img_arr, f):   # input image, direction
    rgb_to_hsv = np.vectorize(colorsys.rgb_to_hsv)
    hsv_to_rgb = np.vectorize(colorsys.hsv_to_rgb)
    arr = img_arr.astype('float')
    r, g, b = np.rollaxis(arr, axis = -1)
    h, s, v = rgb_to_hsv(r, g, b)
    s *= f
    s /= np.max(s)
    r, g, b = hsv_to_rgb(h, s, v)
    new_img = np.dstack((r,g,b))
    return new_img.astype('uint8')

# manipulate brightness
def mani_b(img_arr, f):   # input image, direction
    img = Image.fromarray(img_arr, 'RGB')
    enhancer = ImageEnhance.Brightness(img)
    new_img = enhancer.enhance(f)
    return np.array(new_img)

# manipulate white balance
# pair should be 'rg', 'gb' or 'rb'
def mani_wb(img_arr, pair, f):   # input image, direction, pair (red and green, green and blue or red and blue)
    img = img_arr.astype('float')
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
    a = actionlst()
    new_img = a[0](img, 0.5)

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