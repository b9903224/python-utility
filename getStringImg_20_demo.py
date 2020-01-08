# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import utility
from skimage import transform
import time

plt.close('all')

text = r'''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`~!@#$%^&*()-_=+[{]}\|;:'",<.>/? 我'''
tStart = time.time()
stringImg = utility.getStringImg_20(text)
tEnd = time.time()
print('Elapsed time is %g seconds.'%(tEnd-tStart))
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