# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:59:52 2020

@author: mchsiaoj
"""

import time
from utility import tictoc
import functools

def yell(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs) + '!'
    return wrapper

@tictoc
@yell
def say_hello(sleep_time, user, **kwargs):
    time.sleep(sleep_time)
    return f'Hello {user}'

if __name__ == '__main__':
    print(say_hello(3, 'Dan'))
    
    logtime_data = {}
    print(say_hello(2, 'Jay', log_time=logtime_data))
    print(logtime_data)
    
#    logtime_data = {}
    print(say_hello(1, 'Lin', log_time=logtime_data, log_name='timeit_test'))
    print(logtime_data)