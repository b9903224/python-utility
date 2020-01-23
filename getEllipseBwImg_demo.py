# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 23:04:03 2020

@author: mchsiaoj

insert binary ellipse image on binary image
output: ndarray.bool
translate from Matlab version:
https://github.com/b9903224/Matlab-utility
https://www.mathworks.com/matlabcentral/fileexchange/73150-getellipsebwimg

"""

import numpy as np
from PIL import Image
import os
import utility
import matplotlib.pyplot as plt

plt.close('all')
demoPath = r'.\demo\getEllipseBwImg'
imgH = 512;
imgW = 1024;
ellipseBwImg = []
#ellipseBwImg = np.zeros((imgH, imgW), np.bool);
#ellipseBwImg[50:200, 50:200] = True

if __name__ == '__main__':
    
    ellipseBwImg_matlab = np.array(Image.open(os.path.join(demoPath, 'ellipseBwImg_matlab.png')))
    utility.imshow(ellipseBwImg_matlab, 'ellipseBwImg_matlab')
    
    ellipseBwImg = utility.getEllipseBwImg(imgH, imgW, 256-1, 512-1, 64, 128, 15, ellipseBwImg);
    ellipseBwImg = utility.getEllipseBwImg(imgH, imgW, 128-1, 256-1, 32, 64, -30, ellipseBwImg);
    ellipseBwImg = utility.getEllipseBwImg(imgH, imgW, 256+128-1, 256-1, 32, 64, 30, ellipseBwImg);
    ellipseBwImg = utility.getEllipseBwImg(imgH, imgW, 512-1, 256+512-1, 512, 64, 30, ellipseBwImg);
    ellipseBwImg = utility.getEllipseBwImg(imgH, imgW, 512-1, 256-1, 64, 512, 0, ellipseBwImg); 
    ellipseBwImg = utility.getEllipseBwImg(imgH, imgW, 256-1, 90-1, 128, 64, 0, ellipseBwImg); 
    ellipseBwImg = utility.getEllipseBwImg(imgH, imgW, 87-1, 685-1, 512, 30, -45, ellipseBwImg); 
    
    utility.imshow(ellipseBwImg, 'ellipseBwImg')
#    Image.fromarray(ellipseBwImg).save(os.path.join(demoPath, 'ellipseBwImg_python.png'))
    
    # check version difference between Matlab and python
    ellipseBwImg_version_xor = ellipseBwImg != ellipseBwImg_matlab
    print('difference pixel count between Matlab and python: %g\n pixels'%(ellipseBwImg_version_xor.sum()))
    utility.imshow(ellipseBwImg_version_xor, 'ellipseBwImg_version_xor')



