# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 22:52:51 2020

@author: hp
"""
import re
txt = "123-41412-41"
x = re.findall("[a-z]", txt)
print(x)