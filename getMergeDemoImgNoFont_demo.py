# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import utility
import time
from skimage import io

# config
plt.close('all')
imgH = 60
imgW = 60
kernel_size = 5

demoPath = r'.\demo\getMergeDemoImgNoFont'
dpi = 512

def getBwImgByChartData(imgH, imgW, y1_ary, imgType):
    y1_ary = y1_ary.astype(np.int)
    img1 = np.zeros((imgH, imgW), dtype=np.bool)
    for x in range(imgW):
        img1[imgH-y1_ary[x]:, x] = True
    if imgType == 'rgb':
        img1 = np.dstack((img1,img1,img1)).astype(np.uint8) * 255
    elif imgType == 'gray':
        img1 = img1.astype(np.uint8) * 255
    elif imgType == 'bool':
        pass
    return img1

# generate dummpy data (chart)
y1_ary = np.linspace(5, 50, 60) + np.random.randn(60) * 5
y1_ary_filtered = signal.medfilt(y1_ary, kernel_size=kernel_size)
y2_ary = np.zeros(60) + imgH / 2 + np.random.randn(60) * 5
y2_ary_filtered = signal.medfilt(y2_ary, kernel_size=kernel_size)
y3_ary = np.linspace(50, 5, 60) + np.random.randn(60) * 5
y3_ary_filtered = signal.medfilt(y3_ary, kernel_size=kernel_size)

# plot dummy data
plt.figure(1)
plt.subplot(3,1,1)
plt.plot(y1_ary)
plt.plot(y1_ary_filtered, linestyle='--')
plt.ylim([0, imgH])
plt.title('trend up')
plt.subplot(3,1,2)
plt.plot(y2_ary)
plt.plot(y2_ary_filtered, linestyle='--')
plt.ylim([0, imgH])
plt.title('normal')
plt.subplot(3,1,3)
plt.plot(y3_ary)
plt.plot(y3_ary_filtered, linestyle='--')
plt.ylim([0, imgH])
plt.title('trend down')
plt.savefig(r'%s\%s.png'%(demoPath, 'plot'), dip=dpi)

# generate image: rgb_uint8, gray_uint8, bool
img1 = getBwImgByChartData(imgH, imgW, y1_ary, 'rgb')
img1[...,1] = 255
img1_filtered = getBwImgByChartData(imgH, imgW, y1_ary_filtered, 'rgb')
img1_filtered[...,2] = 255
img2 = getBwImgByChartData(imgH, imgW, y2_ary, 'gray')
img2_filtered = getBwImgByChartData(imgH, imgW, y2_ary_filtered, 'gray')
img3 = getBwImgByChartData(imgH, imgW, y3_ary, 'bool')
img3_filtered = getBwImgByChartData(imgH, imgW, y3_ary_filtered, 'bool')

# show image
plt.figure(2)
index = 1
plt.subplot(2,3,index)
plt.imshow(img1, cmap=plt.cm.gray), plt.title('%g (rgb)'%index)
plt.subplot(2,3,index+3)
plt.imshow(img1_filtered, cmap=plt.cm.gray), plt.title('%g_filtered'%index)
index = 2
plt.subplot(2,3,index)
plt.imshow(img2, cmap=plt.cm.gray), plt.title('%g (gray)'%index)
plt.subplot(2,3,index+3)
plt.imshow(img2_filtered, cmap=plt.cm.gray), plt.title('%g_filtered'%index)
index = 3
plt.subplot(2,3,index)
plt.imshow(img3, cmap=plt.cm.gray), plt.title('%g (bool)'%index)
plt.subplot(2,3,index+3)
plt.imshow(img3_filtered, cmap=plt.cm.gray), plt.title('%g_filtered'%index)
plt.savefig(r'%s\%s.png'%(demoPath, 'subplot_img'), dip=dpi)

rgb = [0,0,255]
rgb_b = [200,-50,0]
rgb_demo = [102, 255, 255]
row_num = 2
col_num = 5
font_size = 20
#font_size = 35
gap_ver = 22
#gap_ver = 40
gap_hoz = 50
descp_height = font_size * 5
imgH = 60
imgW = 60
img_list = []
img_list.append([{'img': img1, 'title': 'v1', 'rgb': [255,0,0], 'rgb_b': rgb_b},
                 {'img': img2, 'title': 'v2', 'rgb': rgb, 'rgb_b': [100,0,-50]},
                 {},
                 {'img': img3, 'title': 'v3', 'rgb': rgb, 'rgb_b': rgb_b},
                 {}])
img_list.append([{'img': img1_filtered, 'title': 'v1.filter', 'rgb': rgb, 'rgb_b': rgb_b},
                 {'img': img2_filtered, 'title': 'v2.filter', 'rgb': rgb, 'rgb_b': rgb_b},
                 {},
                 {'img': img3_filtered, 'title': 'v3.filter', 'rgb': rgb, 'rgb_b': rgb_b},
                 {}])
descp_text = []
descp_text.append([{'text': 'descp1: xx', 'rgb': rgb, 'rgb_b': rgb_b},
                   {'text': 'descp2: oo', 'rgb': rgb, 'rgb_b': rgb_b},
                   {},
                   {},
                   {'text': 'kernel ver.: %s'%('0.0.1'), 'rgb': rgb, 'rgb_b': [100,0,-50]}])
descp_text.append([{'text': 'descp3: xx', 'rgb': rgb, 'rgb_b': rgb_b},
                   {},
                   {'text': 'other info: xx', 'rgb': rgb, 'rgb_b': rgb_b}])
descp_test_start_y = [(imgH + gap_ver) * len(img_list), (imgH + gap_ver) * len(img_list)]
descp_test_start_x = [0 + 5, (imgW + gap_hoz) * 2 + 5]

tStart = time.time()
mergedDemoImg = utility.getMergeDemoImgNoFont(rgb,rgb_b,rgb_demo,
                                              row_num,col_num,font_size,gap_ver,gap_hoz,
                                              descp_height,imgH,imgW,
                                              img_list,descp_text,descp_test_start_y,descp_test_start_x)
tEnd = time.time()
print('Elapsed time is %g seconds.'%(tEnd-tStart))

utility.imshow(mergedDemoImg, 'mergedDemoImg')
io.imsave(r'%s\%s.png'%(demoPath, 'mergedDemoImg'), mergedDemoImg)


















