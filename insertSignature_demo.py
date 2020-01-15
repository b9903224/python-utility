# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 22:20:14 2020

@author: b9903
"""

import matplotlib.pyplot as plt
import utility
from skimage import transform
import numpy as np
from skimage import io
from utility.insertSignature import insertSignature
from matplotlib import cm
import time

plt.close('all')
tStart = time.time()
img = io.imread(r'.\demo\insertSignature\lena_color.gif')[:,:,:3]

sigBackRGB = [255,255,255]
fontSize = 20
#img = insertSignature(img, featVer='0.0.1', kernelVer='0.0.1-1')
#img = insertSignature(img, featVer='0.0.1', kernelVer='0.0.1-1', loc='LD')
#img = insertSignature(img, featVer='0.0.1', kernelVer='0.0.1-1', loc='RD')
#img = insertSignature(img, featVer='0.0.1', kernelVer='0.0.1-1', loc='RU')
#img = insertSignature(img, featVer='0.0.1', kernelVer='0.0.1-1', loc='LU')
#img = insertSignature(img, featVer='0.0.1', kernelVer='0.0.1-1', loc='LD', sigBackRGB=sigBackRGB)
img = insertSignature(img, featVer='0.0.1', kernelVer='0.0.1-1', loc='RD', sigBackRGB=sigBackRGB)
#img = insertSignature(img, featVer='0.0.1', kernelVer='0.0.1-1', loc='RU', sigBackRGB=sigBackRGB)
#img = insertSignature(img, featVer='0.0.1', kernelVer='0.0.1-1', loc='LU', sigBackRGB=sigBackRGB)
img = insertSignature(img, featVer='0.0.1', kernelVer='0.0.1-1', loc='LD', fontSize=fontSize)
tEnd = time.time()
print('Elapsed time is %g seconds.'%(tEnd-tStart))

utility.imshow(img, 'img')
#io.imsave(r'.\demo\insertSignature\lena_color_with_signature.png', img)

# get 256*256 jet colormap
#colors = cm.jet(range(0,256))[:,:3].T
#img = np.dstack((colors[0, :], colors[1, :], colors[2, :]))
#img = np.tile(img, (256, 1, 1)) # = Matlab's repmat
#img_uint8 = (img*255).astype(np.uint8)
#utility.imshow(img, 'img').axis('off')
#utility.imshow(img_uint8, 'img_uint8').axis('off')

# get 256*256 jet colormap: Matlab
#colors = jet(256);
#colors = colors';
#img = cat(3, colors(1, :), colors(2, :), colors(3, :));
#img = repmat(img, [256, 1]);
#img_uint8 = uint8(img * 255);
#figure, imshow(img), title('img')
#figure, imshow(img_uint8), title('img_uint8', 'interpreter', 'none')

