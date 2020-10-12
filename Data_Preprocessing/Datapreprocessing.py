# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 10:23:26 2020

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(x)
print(y)