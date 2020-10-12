# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 12:45:06 2020

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

#Encoding Categorical Data
#OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[0])], remainder = 'passthrough')
x = np.array(ct.fit_transform(x))
print(x)
#LabelEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

#Spliting Dataset into training and test sets
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 0.2, random_state = 1)
print(xtrain)
print(xtest)
print(ytrain)
print(ytest)

#Feature Scale
#Only Apply feature scaling to numerical values rather than dummy variables
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xtrain[:,3:] = sc.fit_transform(xtrain[:,3:])
xtest[:,3:] = sc.transform(xtest[:,3:])
print(xtrain)
print(xtest)

