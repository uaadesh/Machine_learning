'''
1.Importing the libraries 
2.Import the dataset
3.Finding our target and predictor
4.Data preprocessing
5.Train test split
6.Model making
7.Evaluation of the model
8.Plots and Graphs
9.Optimization
Assigment - find the difference between CSV file
and Xcel sheet.
'''
# Simple Linear Regression

# Importing the libraries
import numpy as np 
#all the numerical analysis is done using numpy
import matplotlib.pyplot as plt
#is used for all the plots and graphs
import pandas as pd
#deals with file handeling

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
#Fidnig the independent variable(Predictor)
X = dataset.iloc[:, :1].values
#Finding the dependent variable(Target)
y = dataset.iloc[:, 1].values

#Find out more about vectors and matrix of features

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
#ideal split 80 training:20 testing. 
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# Fitting Simple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict([[6.12]])

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()