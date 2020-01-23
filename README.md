# python-utility
@ Ming-Chih, Hsiao

commonly used package

## [getEllipseBwImg](utility/getEllipseBwImg.py)
insert binary ellipse image on binary image
```python
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
    Image.fromarray(ellipseBwImg).save(os.path.join(demoPath, 'ellipseBwImg_python.png'))
    
    # check version difference between Matlab and python
    ellipseBwImg_version_xor = ellipseBwImg != ellipseBwImg_matlab
    print('difference pixel count between Matlab and python: %g\n pixels'%(ellipseBwImg_version_xor.sum()))
    utility.imshow(ellipseBwImg_version_xor, 'ellipseBwImg_version_xor')
```
<img src="https://github.com/b9903224/python-utility/blob/master/demo/getEllipseBwImg/ellipseBwImg_python.png"/>
    
## [insertTextImgByRange](utility/insertTextImgByRange.py)
```python
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
```
<img src="https://github.com/b9903224/python-utility/blob/master/demo/insertTextImgByRange/img_original.png" width="500px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/insertTextImgByRange/img_processed.png" width="500px" />

## [insertSignature](utility/insertSignature.py)

```python
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
img = insertSignature(img, featVer='0.0.1', kernelVer='0.0.1-1')
#img = insertSignature(img, featVer='0.0.1', kernelVer='0.0.1-1', loc='L')
tEnd = time.time()
print('Elapsed time is %g seconds.'%(tEnd-tStart))

utility.imshow(img, 'img')
io.imsave(r'.\demo\insertSignature\lena_color_with_signature.png', img)


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
```
<img src="https://github.com/b9903224/python-utility/blob/master/demo/insertSignature/lena_color.gif" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/insertSignature/lena_color_with_signature.png" width="250px" />


## [getScaledImg](utility/getScaledImg.py)
```python
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
```
<img src="https://github.com/b9903224/python-utility/blob/master/demo/getScaledImg/fig.png" width="750px" />

## [getMergeDemoImgNoFont](utility/getMergeDemoImgNoFont.py)
```python
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
```
<img src="https://github.com/b9903224/python-utility/blob/master/demo/getMergeDemoImgNoFont/plot.png" width="250px" />
<img src="https://github.com/b9903224/python-utility/blob/master/demo/getMergeDemoImgNoFont/subplot_img.png" width="250px" />
<img src="https://github.com/b9903224/python-utility/blob/master/demo/getMergeDemoImgNoFont/mergedDemoImg.png" width="250px" />

## [getHighLightImgByRange](utility/getHighLightImgByRange.py)

```python
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
```
<img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImgByRange/stringImg.png" width="250px" />
<img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImgByRange/img_rgb.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImgByRange/img_gray.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImgByRange/img_bw.png" width="250px" />

<img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImgByRange/img_rgb_demo.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImgByRange/img_rgb_demo2.png" width="250px" />

<img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImgByRange/img_gray_demo.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImgByRange/img_gray_demo2.png" width="250px" />

<img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImgByRange/img_bw_demo.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImgByRange/img_bw_demo2.png" width="250px" />

## [getStringImg_20](utility/getStringImg_20.py)
Get binary text image not use font file.
```python
import matplotlib.pyplot as plt
import utility
from skimage import transform

plt.close('all')

text = r'''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`~!@#$%^&*()-_=+[{]}\|;:'",<.>/? 我'''
stringImg = utility.getStringImg_20(text)
stringImg2 = transform.rescale(stringImg, 1.5, order=0)
utility.imshow(stringImg)

text = r'Hello World !'
stringImg = utility.getStringImg_20(text)
utility.imshow(stringImg).savefig('.\demo\getStringImg_20\stringImg_20_demo.png', dip=512)
text = r'a'
stringImg = utility.getStringImg_20(text)
utility.imshow(stringImg)
text = r'abcABC: 123'
stringImg = utility.getStringImg_20(text)
utility.imshow(stringImg)
text = r' '
stringImg = utility.getStringImg_20(text)
utility.imshow(stringImg)
```
<img src="https://github.com/b9903224/python-utility/blob/master/demo/getStringImg_20/stringImg_20_demo.png"/>

## [getFontTemplateImg](utility/getFontTemplateImg.py)
Output font image by font file
```python
import utility
from skimage import io
import os
from utility.getFontTemplateImg import getFontTemplateImg

if __name__ == '__main__':
    font_path = r'.\demo\font\EightBitDragon-anqx.ttf'
    imgH = 20 # = font_size
    imgW = 1300
    fontTemplateImg = getFontTemplateImg(imgH, imgW, font_path)
    font_file_name = os.path.splitext(os.path.basename(font_path))[0] # get fileName without extention
    io.imsave(r'.\demo\font_template\font_template_%s.png'%(font_file_name), fontTemplateImg)
    utility.imshow(fontTemplateImg, 'fontTemplateImg')
```
https://www.fontspace.com/chequered-ink/eight-bit-dragon
<img src="https://github.com/b9903224/python-utility/blob/master/demo/font_template/font_template_EightBitDragon-anqx.png"/>
Minecraft-Regular: recommend for 10 times font size, very beautiful 8-bit font, but don't have the character: `
<img src="https://github.com/b9903224/python-utility/blob/master/demo/font_template/font_template_1_Minecraft-Regular_fontSize%3D20.png"/>

## [imshow](utility/imshow.py)
Customized for matploblib.pyplot
```python
import numpy as np
import matplotlib.pyplot as plt
import utility
#import copy

def getImgBw():
    img_bw_false = np.zeros((3, 3), np.bool)
    #img_bw_true = copy.deepcopy(img_bw_false)
    img_bw_true = np.copy(img_bw_false)
    img_bw_true.fill(True)
    img_bw_mix = np.zeros((3, 3), np.bool)
    pos = [(0,0),(1,1),(2,2)]
    rows, cols = zip(*pos)
    img_bw_mix[rows, cols] = True
    return img_bw_false, img_bw_true, img_bw_mix
def getImgGray():
    img_gray_0 = np.zeros((3,3),np.uint8)
    img_gray_127 = np.copy(img_gray_0)
    img_gray_127.fill(127)
    img_gray_255 = np.copy(img_gray_0)
    img_gray_255.fill(255)
    
    img_gray_mix = np.copy(img_gray_0)
    img_gray_mix.fill(127)
    img_gray_mix[(0,2),(0,2)] = [0,255]
    return img_gray_0, img_gray_127, img_gray_255, img_gray_mix
def getImgRgb():
    img0 = np.zeros((3,3),np.uint8)
    img127 = np.copy(img0)
    img127.fill(127)
    img255 = np.copy(img0)
    img255.fill(255)
    img_rgb_black = np.dstack((img0,img0,img0))
    img_rgb_gray = np.dstack((img127,img127,img127))
    img_rgb_white = np.dstack((img255,img255,img255))
    img_rgb_R = np.dstack((img255,img0,img0))
    img_rgb_G = np.dstack((img0,img255,img0))
    img_rgb_B = np.dstack((img0,img0,img255))
    return img_rgb_black,img_rgb_gray,img_rgb_white,img_rgb_R,img_rgb_G,img_rgb_B

plt.close('all')

img_bw_false, img_bw_true, img_bw_mix = getImgBw()
img_gray_0, img_gray_127, img_gray_255, img_gray_mix = getImgGray()
img_rgb_black,img_rgb_gray,img_rgb_white,img_rgb_R,img_rgb_G,img_rgb_B = getImgRgb()

dpi = 128
utility.imshow(img_bw_false, 'img_bw_false').savefig(r'.\demo\imshow\img_bw_false.png', dpi=dpi)
utility.imshow(img_bw_true, 'img_bw_true').savefig(r'.\demo\imshow\img_bw_true.png', dpi=dpi)
utility.imshow(img_bw_mix, 'img_bw_mix').savefig(r'.\demo\imshow\img_bw_mix.png', dpi=dpi)
utility.imshow(img_gray_0, 'img_gray_0').savefig(r'.\demo\imshow\img_gray_0.png', dpi=dpi)
utility.imshow(img_gray_127, 'img_gray_127').savefig(r'.\demo\imshow\img_gray_127.png', dpi=dpi)
utility.imshow(img_gray_255, 'img_gray_255').savefig(r'.\demo\imshow\img_gray_255.png', dpi=dpi)
utility.imshow(img_gray_mix, 'img_gray_mix').savefig(r'.\demo\imshow\img_gray_mix.png', dpi=dpi)
utility.imshow(img_rgb_black, 'img_rgb_black').savefig(r'.\demo\imshow\img_rgb_black.png', dpi=dpi)
utility.imshow(img_rgb_gray, 'img_rgb_gray').savefig(r'.\demo\imshow\img_rgb_gray.png', dpi=dpi)
utility.imshow(img_rgb_white, 'img_rgb_white').savefig(r'.\demo\imshow\img_rgb_white.png', dpi=dpi)
utility.imshow(img_rgb_R, 'img_rgb_R').savefig(r'.\demo\imshow\img_rgb_R.png', dpi=dpi)
utility.imshow(img_rgb_G, 'img_rgb_G').savefig(r'.\demo\imshow\img_rgb_G.png', dpi=dpi)
utility.imshow(img_rgb_B, 'img_rgb_B').savefig(r'.\demo\imshow\img_rgb_B.png', dpi=dpi)
```
<img src="https://github.com/b9903224/python-utility/blob/master/demo/imshow/img_bw_false.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/imshow/img_bw_true.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/imshow/img_bw_mix.png" width="250px" />
<img src="https://github.com/b9903224/python-utility/blob/master/demo/imshow/img_gray_0.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/imshow/img_gray_127.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/imshow/img_gray_255.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/imshow/img_gray_mix.png" width="250px" />

<img src="https://github.com/b9903224/python-utility/blob/master/demo/imshow/img_rgb_black.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/imshow/img_rgb_gray.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/imshow/img_rgb_white.png" width="250px" />
<img src="https://github.com/b9903224/python-utility/blob/master/demo/imshow/img_rgb_R.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/imshow/img_rgb_G.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/imshow/img_rgb_B.png" width="250px" />

## [getHighLightImg](utility/getHighLightImg.py)
```python
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, morphology, color
import utility
#from utility.getHighLightImg import getImgSize

def loadImg():
    img_bw = io.imread(r".\demo\getHighLightImg\img_bw.png") # 1 depth trasform to uint8 [0, 255]
    img_bw = img_bw != 0
    img_gray = io.imread(r".\demo\getHighLightImg\img_gray.png")
    img_rgb = io.imread(r".\demo\getHighLightImg\img_rgb.png")
    return img_bw, img_gray, img_rgb
def getBoundingFilter_bwAndGray(img_bw):
    boundingFilter = morphology.dilation(img_bw, morphology.square(5))
    temImg = morphology.dilation(img_bw, morphology.square(3))
    boundingFilter[temImg] = False
    return boundingFilter
def getBoundingFilter_rgb(img_rgb):
    img_gray = (color.rgb2gray(img_rgb)*255).astype(np.uint8)
    img_bw = img_gray > 200
    oundingFilter_rgb = morphology.dilation(img_bw, morphology.square(7))
    temImg = morphology.dilation(img_bw, morphology.square(3))
#    oundingFilter_rgb[temImg] = False
    return oundingFilter_rgb
if __name__ == '__main__':
    plt.close('all')
    img_bw, img_gray, img_rgb = loadImg()
    boundingFilter_bwAndGray = getBoundingFilter_bwAndGray(img_bw)
    boundingFilter_rgb = getBoundingFilter_rgb(img_rgb)
    img_bw_bounding = utility.getHighLightImg(boundingFilter_bwAndGray, img_bw)
    img_gray_bounding = utility.getHighLightImg(boundingFilter_bwAndGray, img_gray)
    img_rgb_bounding = utility.getHighLightImg(boundingFilter_rgb, img_rgb)
    img_rgb_bounding2 = utility.getHighLightImg(boundingFilter_rgb, img_rgb, rgb=[128, 0, 0], isTransparent=True, rgbBackground=[0,60,0])
    img_rgb_bounding3 = utility.getHighLightImg(boundingFilter_rgb, img_rgb, rgb=[-128, -128, 0], isTransparent=True, rgbBackground=[0,60,0])
    ##

    utility.imshow(img_bw, 'img_bw')
    utility.imshow(img_gray, 'img_gray')
    utility.imshow(img_rgb, 'img_rgb')
    utility.imshow(boundingFilter_bwAndGray, 'boundingFilter_bwAndGray')
    utility.imshow(boundingFilter_rgb, 'boundingFilter_rgb')
    utility.imshow(img_bw_bounding, 'img_bw_bounding')
    utility.imshow(img_gray_bounding, 'img_gray_bounding')
    utility.imshow(img_rgb_bounding, 'img_rgb_bounding')
    utility.imshow(img_rgb_bounding2, 'img_rgb_bounding2')
    utility.imshow(img_rgb_bounding3, 'img_rgb_bounding3')
```
<img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImg/result/img_bw.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImg/result/img_gray.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImg/result/img_rgb.png" width="250px" />
<img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImg/result/boundingFilter_bwAndGray.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImg/result/boundingFilter_rgb.png" width="250px" />

<img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImg/result/img_bw_bounding.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImg/result/img_gray_bounding.png" width="250px" />

<img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImg/result/img_rgb_bounding.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImg/result/img_rgb_bounding2.png" width="250px" /><img src="https://github.com/b9903224/python-utility/blob/master/demo/getHighLightImg/result/img_rgb_bounding3.png" width="250px" />

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

