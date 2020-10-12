# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 16:24:41 2020

@author: hp
"""

#Import The required libraries

# from sklearn import model_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.impute import SimpleImputer
# x = [[0.18],[1.0],[0.92],[0.07],[0.85],[0.99],[0.87]]

dataset = pd.DataFrame( [[0.18 ,0.89 ,109.85],
[1.0 ,0.26, 155.72],
[0.92, 0.11, 137.66],
[0.07, 0.37 ,76.17],
[0.85, 0.16, 139.75],
[0.99, 0.41 ,162.6],
[0.87, 0.47, 151.77]])
datas = pd.DataFrame([[0.49,0.18],
[0.57,0.83],
[0.56,0.64],
[0.76,0.18]])
# print(dataset)
#Load the data
# dataset = pd.read_csv('50_Startups.csv')
# print(dataset)
# #Independent Variables
x = dataset.iloc[:,:-1].values

# #Dependent Variables
y = dataset.iloc[:,-1].values
print(y)


# #Encoding Categorical data
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import OneHotEncoder
# ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[3])], remainder = 'passthrough')
# x = np.array(ct.fit_transform(x))

# print(x)

# #Splitting into train and test
# from sklearn.model_selection import train_test_split
# xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2,random_state = 0)

# #Training the model 
from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()
Regressor.fit(x,y)

# #Predicting the results
ypred = Regressor.predict(datas)
print(ypred)
# np.set_printoptions(precision=2)
# print(np.concatenate((ypred.reshape(len(ypred),1),ytest.reshape(len(ytest),1)),1))


