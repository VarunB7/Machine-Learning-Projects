# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 20:52:17 2020

@author: hp
"""

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 

#import dataset
dataset = pd.read_csv('Position_Salaries.csv')

x = dataset.iloc[:,1:2].values
print(x)


y = dataset.iloc[:,-1].values
print(y)

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree = 4)
xpoly = polyreg.fit_transform(x)
regressor = LinearRegression()
regressor.fit(xpoly,y)
ypred = regressor.predict(xpoly)

plt.scatter(x,y)
plt.plot(x,ypred,color ='red')
