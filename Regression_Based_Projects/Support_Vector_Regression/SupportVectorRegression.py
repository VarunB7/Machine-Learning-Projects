# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 21:05:33 2020

@author: hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r'C:\Users\hp\Desktop\MLDL\Machine Learning A-Z (Model Selection)\Regression\Data.csv')

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest =train_test_split(x,y, test_size =0.2, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
Sc_x = StandardScaler()
Sc_y = StandardScaler()
ytrain = ytrain.reshape(len(ytrain),1)
ytrain = Sc_y.fit_transform(ytrain)
xtrain = Sc_x.fit_transform(xtrain)

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(xtrain,ytrain)                                                                                                                                                      


ypred = Sc_y.inverse_transform(regressor.predict(Sc_x.fit_transform(xtest)))
print(ypred)

from sklearn.metrics import r2_score
score = r2_score(ytest,ypred)
print(score)



# plt.scatter(Sc_x.inverse_transform(x),Sc_y.inverse_transform(y),color ='red')
# plt.plot(Sc_x.inverse_transform(x),Sc_y.inverse_transform(regressor.predict(x)),color = 'blue')
# plt.title('SVR on Pos_Sal')
# plt.xlabel('Postion')
# plt.ylabel('Salaries')
# plt.show()

#Smoother Curve representation
# xgrid = np.arange(min(Sc_x.inverse_transform(x)),max(Sc_x.inverse_transform(x)),0.1)
# xgrid = xgrid.reshape(len(xgrid),1)

# plt.scatter(Sc_x.inverse_transform(x),Sc_y.inverse_transform(y),color ='red')
# plt.plot(xgrid,Sc_y.inverse_transform(regressor.predict(xgrid)),color = 'blue')
# plt.title('SVR on Pos_Sal')
# plt.xlabel('Postion')
# plt.ylabel('Salaries')
# plt.show()
