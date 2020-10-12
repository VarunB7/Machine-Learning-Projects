# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 16:54:33 2020

@author: hp
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Importin the dataset
dataset = pd.read_csv(r'C:\Users\hp\Desktop\MLDL\Machine Learning A-Z (Model Selection)\Regression\Data.csv') 
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest =train_test_split(x,y, test_size =0.2, random_state = 0)

#Train the Model using data 
from sklearn.linear_model import LinearRegression
# Lin_regressor = LinearRegression()
# Lin_regressor.fit(xtrain,ytrain)

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
xpoly = poly_reg.fit_transform(xtrain)
Lin_regressor2 = LinearRegression()
Lin_regressor2.fit(xpoly,ytrain)

# plt.scatter(x,y,color = 'red')
# plt.plot(xtrain,Lin_regressor.predict(xtrain),color='blue')
# plt.show()

# plt.scatter(xtrain,ytrain,color = 'red')
# plt.plot(xtest,Lin_regressor2.predict(xpoly),color='blue')
# plt.show()

ypred = Lin_regressor2.predict(poly_reg.transform(xtest))


from sklearn.metrics import r2_score
score = r2_score(ytest,ypred)
print(score)