# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 12:07:36 2020

@author: hp
"""


#Import The required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

#Load the data
dataset = pd.read_csv('Data.csv')
#Independent Variables
x = dataset.iloc[:,:-1].values
#Dependent Variables
y = dataset.iloc[:,-1].values
print(x)
print(y)

#Fill the missing data with the mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
print(x)
print(y)

