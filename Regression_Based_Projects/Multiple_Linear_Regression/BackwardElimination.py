# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 18:43:53 2020

@author: hp
"""

#Import The required libraries

# from sklearn import model_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.impute import SimpleImputer

#Load the data
dataset = pd.read_csv('50_Startups.csv')
#Independent Variables
x = dataset.iloc[:,:-1].values

#Dependent Variables
y = dataset.iloc[:,-1].values


#Encoding Categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[3])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))


#Splitting into train and test
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 0)

#Training the model 
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(xtrain,ytrain)

#Predicting the results
ypred = Regressor.predict(xtest)
# np.set_printoptions(precision=2)
# print(np.concatenate((ypred.reshape(len(ypred),1),ytest.reshape(len(ytest),1)),1))

import statsmodels.api as sm
x = np.append(arr =  np.ones((50,1)).astype(int),values =x,axis =1)
xopt = floatx[:,[0,1,2,3,4,5]]
regressorOLS =sm.OLS(endog = y,exog = xopt).fit()
