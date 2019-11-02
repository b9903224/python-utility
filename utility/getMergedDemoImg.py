# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:39:15 2019

@author: b9903
"""

import numpy as np

def getMergedDemoImg(row_num, col_num, imgH, imgW,
                     img_list, title_list, gap_ver, gap_hoz,
                     font_path, font_size, font_rgb, font_background_rgb,
                     demo_background_rgb, append_height, descp_text,
                     descp_test_start_y, descp_test_start_x):
    
    mergedDemoImg = iniMergedDemoImg(row_num * (imgH + gap_ver) + append_height,
                                     col_num * (imgW + gap_hoz), demo_background_rgb)
    mergedDemoImg = insert_img_list(mergedDemoImg, img_list, imgH, imgW, gap_ver, gap_hoz)
#    mergedDemoImg = insert_title_list(mergedDemoImg)
            
    
    
    return mergedDemoImg

def insert_img_list(mergedDemoImg, img_list, imgH, imgW, gap_ver, gap_hoz):
    
    for row_num, img_row in enumerate(img_list):
        y = row_num * imgH + (row_num + 1) * gap_ver
        
        for col_num, img in enumerate(img_row):
            if len(img) == 0:
                continue
            
            x = col_num * (imgW + gap_hoz)
            
            if img.ndim == 2:
                if img.dtype == np.bool:
                    img = img.astype(np.uint8) * 255
                    for ch in range(3):
                        mergedDemoImg[y:y+imgH, x:x+imgW, ch] = img
                if img.dtype == np.uint8:
                    for ch in range(3):
                        mergedDemoImg[y:y+imgH, x:x+imgW, ch] = img
            elif img.ndim == 3:
                for ch in range(3):
                    mergedDemoImg[y:y+imgH, x:x+imgW, ch] = img[...,ch]
                
                
                    
    return mergedDemoImg
            
            

def iniMergedDemoImg(imgH, imgW, demo_background_rgb):
    
    mergedDemoImg = np.zeros((imgH, imgW, 3), dtype=np.uint8)
    for ch in range(3):
        mergedDemoImg[..., ch] = demo_background_rgb[ch]
        
    return mergedDemoImg




