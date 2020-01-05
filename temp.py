# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt

def imshow(img, imgName='img', isFullScreen=False):
    plt.figure()
    plt.imshow(img, cmap=plt.cm.gray)
    plt.title('%s: (%gX%gX%g)\ndtype:%s, max:%g, min:%g'%(imgName, *getImgSize(img), img.dtype, img.max(), img.min()))
    if isFullScreen:
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
def getImgSize(img):
    imgZ = 0
    imgH = img.shape[0]
    imgW = img.shape[1]
    if img.ndim == 3:
        imgZ = img.shape[2]
    return imgH, imgW, imgZ

plt.close('all')
img_duck = cv2.imread(r'D:\Users\b9903\Desktop\Python\duck.jpg')
img_duck = img_duck[...,::-1]
img_duck2 = cv2.add(img_duck, 255)

imshow(img_duck,'img_duck')
imshow(img_duck2,'img_duck2')