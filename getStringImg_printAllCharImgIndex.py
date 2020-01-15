# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
import utility
from skimage import io
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import random
from utility.getFontTemplateImg import getFontTemplateImg

text = r'''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`~!@#$%^&*()-_=+[{]}\|;:'",<.>/? '''

def getImgRGBSample(imgH, imgW):
    img = np.zeros((imgH, imgW), np.uint8)
    img.fill(0)
    img = np.dstack((img,img,img))
    return img
def getListString(input):
    output = ','.join(map(str, input))
    output = '[' + output + ']'
    return output
if __name__ == '__main__':
#    font_path = r'.\demo\font\EightBitDragon-anqx.ttf'
    font_path = r'.\demo\font\1_Minecraft-Regular.otf'
    font_size = 10
    font_rgb = [255,255,255]
    font = ImageFont.truetype(font_path, font_size)
    imgH = font_size # = font_size
    imgW = font_size
#    y_ori = 0
    y_ori = -2
    img_char_th = 90
    img_0 = getImgRGBSample(imgH, imgW)

    print('if False:\n    pass')
    for index, char in enumerate(text):
        img = np.copy(img_0)
        frame = Image.fromarray(img)
        draw = ImageDraw.Draw(frame)
        draw.text((1,y_ori), char, tuple(font_rgb), font=font)
        img_char = np.array(frame)
        img_char = img_char[:,:,0]
        rows, cols = np.nonzero(img_char > img_char_th)
        print('elif char == \'%s\':'%(char))
        print('%srows = %s'%(' '*4, getListString(rows)))
        print('%scols = %s'%(' '*4, getListString(cols)))
#        print('%simg[rows,cols]=True'%(' '*4))
    
    
#a = np.array([[0,1,0], [1,1,0], [1,0,0]], np.bool)
#rows, cols = np.nonzero(a)
#b = np.zeros(a.shape, np.bool)
#b[[0,1,1,2],[1,0,1,0]] = True
#utility.imshow(a)
#utility.imshow(b)

#a = np.array([[random.choice([0, 1]),random.choice([0, 1]),random.choice([0, 1])],[random.choice([0, 1]),random.choice([0, 1]),random.choice([0, 1])],[random.choice([0, 1]),random.choice([0, 1]),random.choice([0, 1])]], np.bool)
#b = np.array([[random.choice([0, 1]),random.choice([0, 1]),random.choice([0, 1])],[random.choice([0, 1]),random.choice([0, 1]),random.choice([0, 1])],[random.choice([0, 1]),random.choice([0, 1]),random.choice([0, 1])]], np.bool)
#L = []
#L.append(a)
#L.append(b)
#result = np.concatenate(L, axis=1)
#utility.imshow(a)
#utility.imshow(b)
#utility.imshow(result)










