# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 00:16:40 2020

@author: b9903
"""

import numpy as np

def getHighLightImgByRange(destImg, sourceImg, yStart, xStart, rgb=[255, 0, 0], isTransparent=False, rgbBackground=[]):
    resultImg = np.copy(sourceImg)
    return resultImg