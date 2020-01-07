# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import utility
from skimage import io
from skimage.color import rgb2gray

plt.close('all')

text = r'Hello World !'
stringImg = utility.getStringImg_20(text)

img_rgb = io.imread('.\demo\getHighLightImgByRange\duck.jpg')
img_gray = (rgb2gray(img_rgb) * 255).astype(np.uint8)
img_bw = img_gray < 250

yStart = 200
xStart = 70
img_rgb_demo = utility.getHighLightImgByRange(stringImg, img_rgb, yStart, xStart)
img_rgb_demo2 = utility.getHighLightImgByRange(stringImg, img_rgb, yStart, xStart, rgb=[0, 0, 50], isTransparent=True, rgbBackground=[50,50,0])
img_gray_demo = utility.getHighLightImgByRange(stringImg, img_gray, yStart, xStart)
img_gray_demo2 = utility.getHighLightImgByRange(stringImg, img_gray, yStart, xStart, rgb=[0, 0, 50], isTransparent=True, rgbBackground=[50,50,0])
img_bw_demo = utility.getHighLightImgByRange(stringImg, img_bw, yStart, xStart)
img_bw_demo2 = utility.getHighLightImgByRange(stringImg, img_bw, yStart, xStart, rgb=[0, 0, 50], isTransparent=True, rgbBackground=[50,50,0])

utility.imshow(stringImg)
utility.imshow(img_rgb)
utility.imshow(img_gray)
utility.imshow(img_bw)

utility.imshow(img_rgb_demo)
utility.imshow(img_rgb_demo2)
utility.imshow(img_gray_demo)
utility.imshow(img_gray_demo2)
utility.imshow(img_bw_demo)
utility.imshow(img_bw_demo2)




