# python-utility
commonly used package

## [getMergedDemoImg](utility/getMergedDemoImg.py)
```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import utility
from PIL import Image

# config
plt.close('all')
imgH = 60
imgW = 60
kernel_size = 5

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
plt.savefig('plot.png', dpi=512)
```
![image](https://github.com/b9903224/python-utility/blob/master/demo/getMergedDemoImg/plot.png)
```python

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
plt.savefig('subplot_img.png', dpi=512)
```
![image](https://github.com/b9903224/python-utility/blob/master/demo/getMergedDemoImg/subplot_img.png)

```python
# prepare data
row_num = 2
col_num = 5
imgH = 60
imgW = 60
img_list = [[img1, img2, [], img3, []], [img1_filtered, img2_filtered, [], img3_filtered, []]]
title_list = [['v1', 'v2', [], 'v3'], ['v1.filter', 'v2.filter', [], 'v3.filter', []]]
gap_ver = 22
gap_hoz = 25
#font_path = ''
font_path = r'.\demo\font\1_Minecraft-Regular.otf'
#font_path = r'.\demo\font\pixelmix-1.ttf'
font_size = 20
font_rgb = [0, 0, 255]
font_background_rgb = [0, 200, 0]
demo_background_rgb = [102, 255, 255]
append_height = font_size * 5
descp_text = [['prob_v1: xx', 'prob_v2: oo', '', '', 'kernel ver.: %s'%('1.6.1')],
               ['prob_v3: xx', '', 'other info: xx']]
descp_test_start_y = [(imgH + gap_ver) * 2, (imgH + gap_ver) * 2]
descp_test_start_x = [0 + 5, (imgW + gap_hoz) * 2 + 5]



mergedDemoImg = utility.getMergedDemoImg(row_num, col_num, imgH, imgW,
                                         img_list, title_list, gap_ver, gap_hoz,
                                         font_path, font_size, font_rgb, font_background_rgb,
                                         demo_background_rgb, append_height, descp_text,
                                         descp_test_start_y, descp_test_start_x)
plt.figure(3)
plt.imshow(mergedDemoImg), plt.title('mergedDemoImg')
Image.fromarray(mergedDemoImg).save(r'.\mergedDemoImg.png')
```
![image](https://github.com/b9903224/python-utility/blob/master/demo/getMergedDemoImg/mergedDemoImg.png)


## [printUniqueRatio](utility/printUniqueRatio.py)
```python
import random
import utility

list_len = 10000
dest = ['剪刀', '石頭', '布', True, 1, 2, 3]

list_dest = [random.choice(dest) for _ in range(list_len)]
utility.printUniqueRatio(list_dest)

>>
index: 0 - 1: 28.73%(2873/10000)
index: 1 - 2: 13.64%(1364/10000)
index: 2 - 3: 14.67%(1467/10000)
index: 3 - 布: 14.58%(1458/10000)
index: 4 - 剪刀: 14.28%(1428/10000)
index: 5 - 石頭: 14.10%(1410/10000)
```

