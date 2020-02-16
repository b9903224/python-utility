# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:52:21 2020

@author: mchsiaoj
"""

import json
#import re
from utility.exception_dec import exception_dec

@exception_dec
def run(numerator, denominator):
    result = numerator / denominator
    return result

if __name__ == '__main__':
    result = run(1, 2) # result = {'result': 0.5, 'errMsg': ''}
    result2 = run(1, 0) # result = {'result': None, 'errMsg': '...ZeroDivisionError: division by zero'}
    result3 = run(2, 0) # result = {'result': None, 'errMsg': '...ZeroDivisionError: division by zero'}
    
    result_json = json.dumps(result)
    result2_json = json.dumps(result2)
    result3_json = json.dumps(result3)
    
#    result2['errMsg'] = re.sub('\s+', ' ', result2['errMsg']) # remove space character
#    result3['errMsg'] = " ".join(result3['errMsg'].split()) # remove space character