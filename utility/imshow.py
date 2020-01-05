# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 15:57:08 2020

@author: b9903
"""

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