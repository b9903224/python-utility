# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 14:07:14 2020

@author: b9903
"""

import matplotlib.pyplot as plt
import utility
from skimage import io
#import numpy as np

img_gray = io.imread(r'.\demo\getScaledImg\lenna_low.png')
img_rgb = io.imread(r'.\demo\getScaledImg\office_1.jpg')
img_gray_scaled = utility.getScaledImg(img_gray)
img_rgb_scaled = utility.getScaledImg(img_rgb)

plt.close('all')
plt.subplot(2,2,1)
plt.imshow(img_gray, cmap=plt.cm.gray, vmin=0, vmax=255), plt.title('img_gray')
plt.subplot(2,2,2)
plt.imshow(img_gray_scaled, cmap=plt.cm.gray, vmin=0, vmax=255), plt.title('img_gray_scaled')
plt.subplot(2,2,3)
plt.imshow(img_rgb, cmap=plt.cm.gray, vmin=0, vmax=255), plt.title('img_rgb')
plt.subplot(2,2,4)
plt.imshow(img_rgb_scaled, cmap=plt.cm.gray, vmin=0, vmax=255), plt.title('img_rgb_scaled')
plt.savefig(r'.\demo\getScaledImg\fig.png', dpi=512)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()

#utility.imshow(img_gray)
#utility.imshow(img_gray_scaled)
#utility.imshow(img_rgb)
#utility.imshow(img_rgb_scaled)