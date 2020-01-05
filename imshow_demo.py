# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 22:04:53 2020

@author: b9903
"""

import numpy as np
import matplotlib.pyplot as plt
import utility
#import copy

def getImgBw():
    img_bw_false = np.zeros((3, 3), np.bool)
    #img_bw_true = copy.deepcopy(img_bw_false)
    img_bw_true = np.copy(img_bw_false)
    img_bw_true.fill(True)
    img_bw_mix = np.zeros((3, 3), np.bool)
    pos = [(0,0),(1,1),(2,2)]
    rows, cols = zip(*pos)
    img_bw_mix[rows, cols] = True
    return img_bw_false, img_bw_true, img_bw_mix
def getImgGray():
    img_gray_0 = np.zeros((3,3),np.uint8)
    img_gray_127 = np.copy(img_gray_0)
    img_gray_127.fill(127)
    img_gray_255 = np.copy(img_gray_0)
    img_gray_255.fill(255)
    
    img_gray_mix = np.copy(img_gray_0)
    img_gray_mix.fill(127)
    img_gray_mix[(0,2),(0,2)] = [0,255]
    return img_gray_0, img_gray_127, img_gray_255, img_gray_mix
def getImgRgb():
    img0 = np.zeros((3,3),np.uint8)
    img127 = np.copy(img0)
    img127.fill(127)
    img255 = np.copy(img0)
    img255.fill(255)
    img_rgb_black = np.dstack((img0,img0,img0))
    img_rgb_gray = np.dstack((img127,img127,img127))
    img_rgb_white = np.dstack((img255,img255,img255))
    img_rgb_R = np.dstack((img255,img0,img0))
    img_rgb_G = np.dstack((img0,img255,img0))
    img_rgb_B = np.dstack((img0,img0,img255))
    return img_rgb_black,img_rgb_gray,img_rgb_white,img_rgb_R,img_rgb_G,img_rgb_B

plt.close('all')

img_bw_false, img_bw_true, img_bw_mix = getImgBw()
img_gray_0, img_gray_127, img_gray_255, img_gray_mix = getImgGray()
img_rgb_black,img_rgb_gray,img_rgb_white,img_rgb_R,img_rgb_G,img_rgb_B = getImgRgb()

dpi = 128
utility.imshow(img_bw_false, 'img_bw_false').savefig(r'.\demo\imshow\img_bw_false.png', dpi=dpi)
utility.imshow(img_bw_true, 'img_bw_true').savefig(r'.\demo\imshow\img_bw_true.png', dpi=dpi)
utility.imshow(img_bw_mix, 'img_bw_mix').savefig(r'.\demo\imshow\img_bw_mix.png', dpi=dpi)
utility.imshow(img_gray_0, 'img_gray_0').savefig(r'.\demo\imshow\img_gray_0.png', dpi=dpi)
utility.imshow(img_gray_127, 'img_gray_127').savefig(r'.\demo\imshow\img_gray_127.png', dpi=dpi)
utility.imshow(img_gray_255, 'img_gray_255').savefig(r'.\demo\imshow\img_gray_255.png', dpi=dpi)
utility.imshow(img_gray_mix, 'img_gray_mix').savefig(r'.\demo\imshow\img_gray_mix.png', dpi=dpi)
utility.imshow(img_rgb_black, 'img_rgb_black').savefig(r'.\demo\imshow\img_rgb_black.png', dpi=dpi)
utility.imshow(img_rgb_gray, 'img_rgb_gray').savefig(r'.\demo\imshow\img_rgb_gray.png', dpi=dpi)
utility.imshow(img_rgb_white, 'img_rgb_white').savefig(r'.\demo\imshow\img_rgb_white.png', dpi=dpi)
utility.imshow(img_rgb_R, 'img_rgb_R').savefig(r'.\demo\imshow\img_rgb_R.png', dpi=dpi)
utility.imshow(img_rgb_G, 'img_rgb_G').savefig(r'.\demo\imshow\img_rgb_G.png', dpi=dpi)
utility.imshow(img_rgb_B, 'img_rgb_B').savefig(r'.\demo\imshow\img_rgb_B.png', dpi=dpi)
#












