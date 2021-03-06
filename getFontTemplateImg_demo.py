# -*- coding: utf-8 -*-

import utility
from skimage import io
import os
from utility.getFontTemplateImg import getFontTemplateImg
import matplotlib.pyplot as plt

if __name__ == '__main__':
    plt.close('all')
#    font_path = r'.\demo\font\EightBitDragon-anqx.ttf'
    font_path = r'.\demo\font\1_Minecraft-Regular.otf'
    imgH = 20 # = font_size
    imgW = 1300
#    y_ori = 0
    y_ori = -2
    fontTemplateImg = getFontTemplateImg(imgH, imgW, font_path, y_ori=y_ori)
    font_file_name = os.path.splitext(os.path.basename(font_path))[0] # get fileName without extention
    io.imsave(r'.\demo\font_template\font_template_%s.png'%(font_file_name), fontTemplateImg)
    utility.imshow(fontTemplateImg, 'fontTemplateImg')
#a = np.array([[0,1,0], [1,1,0], [1,0,0]], np.bool)
#rows, cols = np.nonzero(a)
#b = np.zeros(a.shape, np.bool)
#b[rows, cols] = True