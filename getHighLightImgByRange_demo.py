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
img_bw = img_gray > 250

yStart = 100
xStart = 80
#yStart = 340
#xStart = 330
#yStart = 335
#xStart = 4
#yStart = 4
#xStart = 335
img_rgb_demo = utility.getHighLightImgByRange(stringImg, img_rgb, yStart, xStart)
img_rgb_demo2 = utility.getHighLightImgByRange(stringImg, img_rgb, yStart, xStart, rgb=[0, 0, 50], isTransparent=True, rgbBackground=[50,50,0])
img_gray_demo = utility.getHighLightImgByRange(stringImg, img_gray, yStart, xStart)
img_gray_demo2 = utility.getHighLightImgByRange(stringImg, img_gray, yStart, xStart, rgb=[0, 0, 50], isTransparent=True, rgbBackground=[50,50,0])
img_bw_demo = utility.getHighLightImgByRange(stringImg, img_bw, yStart, xStart)
img_bw_demo2 = utility.getHighLightImgByRange(stringImg, img_bw, yStart, xStart, rgb=[0, 0, 50], isTransparent=True, rgbBackground=[50,50,0])

path = r'.\demo\getHighLightImgByRange'
dpi = 512
utility.imshow(stringImg).savefig(r'%s\%s.png'%(path, 'stringImg'), dip=dpi)
utility.imshow(img_rgb).savefig(r'%s\%s.png'%(path, 'img_rgb'), dip=dpi)
utility.imshow(img_gray).savefig(r'%s\%s.png'%(path, 'img_gray'), dip=dpi)
utility.imshow(img_bw).savefig(r'%s\%s.png'%(path, 'img_bw'), dip=dpi)

utility.imshow(img_rgb_demo)
utility.imshow(img_rgb_demo2)
utility.imshow(img_gray_demo)
utility.imshow(img_gray_demo2)
utility.imshow(img_bw_demo)
utility.imshow(img_bw_demo2)

io.imsave(r'%s\%s.png'%(path, 'img_rgb_demo'), img_rgb_demo)
io.imsave(r'%s\%s.png'%(path, 'img_rgb_demo2'), img_rgb_demo2)
io.imsave(r'%s\%s.png'%(path, 'img_gray_demo'), img_gray_demo)
io.imsave(r'%s\%s.png'%(path, 'img_gray_demo2'), img_gray_demo2)
io.imsave(r'%s\%s.png'%(path, 'img_bw_demo'), img_bw_demo)
io.imsave(r'%s\%s.png'%(path, 'img_bw_demo2'), img_bw_demo2)

