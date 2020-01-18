# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 00:06:18 2020

@author: mchsiaoj
"""

import numpy as np
import matplotlib.pyplot as plt
import utility
from skimage import io
from skimage.color import rgb2gray
import time
import os

plt.close('all')
textInfoList = [{'text': 'I', 'rgb': [200,0,0],'rgb_b': [], 'isTransparent': False, 'fontSize': 30, 'yStart': 50, 'xStart': 50},
                 {'text': 'very', 'rgb': [255,255,255],'rgb_b': [0,0,255], 'isTransparent': True, 'fontSize': 10, 'yStart': 80, 'xStart': 80},
                 {'text': 'very', 'rgb': [200,200,0],'rgb_b': [], 'isTransparent': True, 'fontSize': 25, 'yStart': 110, 'xStart': 110},
                 {'text': 'love', 'rgb': [0,0,200],'rgb_b': [200,200,0], 'isTransparent': False, 'fontSize': 40, 'yStart': 140, 'xStart': 140},
                 {'text': 'duck', 'rgb': [-50,0,-50],'rgb_b': [0,-500,-50], 'isTransparent': True, 'fontSize': 50, 'yStart': 170, 'xStart': 170}]

img = io.imread('.\demo\insertTextImgByRange\duck.jpg')
#print(img.shape)

tStart = time.time()
img_processed = utility.insertTextImgByRange(img, textInfoList)
tEnd = time.time()
print('Elapsed time is %g seconds.'%(tEnd-tStart))

path = r'.\demo\insertTextImgByRange'
dpi = 512
utility.imshow(img, 'original').savefig(os.path.join(path, 'img_original.png'), dpi=dpi)
utility.imshow(img_processed, 'processed').savefig(os.path.join(path, 'img_processed.png'), dpi=dpi)