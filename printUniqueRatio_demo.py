# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 14:54:28 2019

@author: b9903
"""

import random
import utility

list_len = 10000
dest = ['剪刀', '石頭', '布', True, 1, 2, 3]

list_dest = [random.choice(dest) for _ in range(list_len)]
utility.printUniqueRatio(list_dest)