# -*- coding: utf-8 -*-
"""
Created on Sat Jan  4 22:54:40 2020

@author: b9903
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, color
import utility
#from utility.getHighLightImg import getImgSize

def loadImg():
    img_bw = io.imread(r".\demo\getHighLightImg\img_bw.png") # 1 depth trasform to uint8 [0, 255]
    img_bw = img_bw != 0
    img_gray = io.imread(r".\demo\getHighLightImg\img_gray.png")
    img_rgb = io.imread(r".\demo\getHighLightImg\img_rgb.png")
    return img_bw, img_gray, img_rgb
def getBoundingFilter_bwAndGray(img_bw):
    boundingFilter = morphology.dilation(img_bw, morphology.square(5))
    temImg = morphology.dilation(img_bw, morphology.square(3))
    boundingFilter[temImg] = False
    return boundingFilter
def getBoundingFilter_rgb(img_rgb):
    img_gray = (color.rgb2gray(img_rgb)*255).astype(np.uint8)
    img_bw = img_gray > 200
    oundingFilter_rgb = morphology.dilation(img_bw, morphology.square(7))
    temImg = morphology.dilation(img_bw, morphology.square(3))
#    oundingFilter_rgb[temImg] = False
    return oundingFilter_rgb
if __name__ == '__main__':
    plt.close('all')
    img_bw, img_gray, img_rgb = loadImg()
    boundingFilter_bwAndGray = getBoundingFilter_bwAndGray(img_bw)
    boundingFilter_rgb = getBoundingFilter_rgb(img_rgb)
    img_bw_bounding = utility.getHighLightImg(boundingFilter_bwAndGray, img_bw)
    img_gray_bounding = utility.getHighLightImg(boundingFilter_bwAndGray, img_gray)
    img_rgb_bounding = utility.getHighLightImg(boundingFilter_rgb, img_rgb)
    img_rgb_bounding2 = utility.getHighLightImg(boundingFilter_rgb, img_rgb, rgb=[128, 0, 0], isTransparent=True, rgbBackground=[0,60,0])
    img_rgb_bounding3 = utility.getHighLightImg(boundingFilter_rgb, img_rgb, rgb=[-128, -128, 0], isTransparent=True, rgbBackground=[0,60,0])
    ##

    utility.imshow(img_bw, 'img_bw')
    utility.imshow(img_gray, 'img_gray')
    utility.imshow(img_rgb, 'img_rgb')
    utility.imshow(boundingFilter_bwAndGray, 'boundingFilter_bwAndGray')
    utility.imshow(boundingFilter_rgb, 'boundingFilter_rgb')
    utility.imshow(img_bw_bounding, 'img_bw_bounding')
    utility.imshow(img_gray_bounding, 'img_gray_bounding')
    utility.imshow(img_rgb_bounding, 'img_rgb_bounding')
    utility.imshow(img_rgb_bounding2, 'img_rgb_bounding2')
    utility.imshow(img_rgb_bounding3, 'img_rgb_bounding3')
    
##
#plt.close('all')
#img_gray = io.imread(r"./demo/getHighLightImg/_blobs_3d_fiji_skeleton.tif")
#img_gray = img_gray[0, :, :]
#img_bw = img_gray == 255
#img_rgb = io.imread(r"./demo/getHighLightImg/rocket.jpg")
#
#plt.figure()
#plt.imshow(img_gray, cmap=plt.cm.gray)
#plt.title('img_gray: (%gX%gX%g)\ndtype:%s, max:%g, min:%g'%(*getImgSize(img_gray), img_gray.dtype, img_gray.max(), img_gray.min()))
#plt.figure()
#plt.imshow(img_bw, cmap=plt.cm.gray)
#plt.title('img_bw: (%gX%gX%g)\ndtype:%s, max:%g, min:%g'%(*getImgSize(img_bw), img_bw.dtype, img_bw.max(), img_bw.min()))
#plt.figure()
#plt.imshow(img_rgb, cmap=plt.cm.gray)
#plt.title('img_rgb: (%gX%gX%g)\ndtype:%s, max:%g, min:%g'%(*getImgSize(img_rgb), img_rgb.dtype, img_rgb.max(), img_rgb.min()))
#
#io.imsave('img_gray.png', img_gray)
##io.imsave('img_bw.png', img_bw) # skimage seems cannot save boolean image to 1 bit depth
## and if imread 1 bit depth image the result will be uint8 [0, 255] numpy array
#io.imsave('img_rgb.png', img_rgb)