# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 10:30:56 2020

@author: hp
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 12:45:06 2020

@author: hp
"""


#Import The required libraries

from sklearn import model_selection
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.impute import SimpleImputer

#Load the data
dataset = pd.read_csv('Salary_Data.csv')
#Independent Variables
x = dataset.iloc[:,:-1].values
#Dependent Variables
y = dataset.iloc[:,-1].values

print(x)
print(y)
#Fill the missing data with the mean
#imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
#imputer.fit(x[:,1:3])
#x[:,1:3] = imputer.transform(x[:,1:3])


#Encoding Categorical Data
#OneHotEncoder
#from sklearn.compose import ColumnTransformer
#from sklearn.preprocessing import OneHotEncoder
#ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(),[0])], remainder = 'passthrough')
#x = np.array(ct.fit_transform(x))

#LabelEncoder
#from sklearn.preprocessing import LabelEncoder
#le = LabelEncoder()
#y = le.fit_transform(y)

#Spliting Dataset into training and test sets

xtrain,xtest,ytrain,ytest = model_selection.train_test_split(x,y,test_size = 0.2, random_state = 1)
print(xtrain)
print(xtest)
print(ytrain)
print(ytest)

#Feature Scale
#Only Apply feature scaling to numerical values rather than dummy variables
#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#xtrain[:,3:] = sc.fit_transform(xtrain[:,3:])
#xtest[:,3:] = sc.transform(xtest[:,3:])
#print(xtrain)
#print(xtest)

#Apply linear regression    
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain,ytrain)

ypred = regressor.predict(xtest)
print(ypred)

#Visualizing the training set result
plt.scatter(xtrain,ytrain,color = 'red')
plt.plot(xtrain,regressor.predict(xtrain), color = 'blue')
plt.title('Salary vs Experience(Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#Visualizing the test set result
plt.scatter(xtest,ytest,color = 'red')
plt.plot(xtrain,regressor.predict(xtrain), color = 'blue')
plt.title('Salary vs Experience(Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

ypred2 = regressor.predict([[5.5],[6.9]])
print(ypred2)