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
Assigment - find about dummy variable trap in the machine learning
perspective.
find out about list
'''
# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
y = dataset['Profit'].values
dataset = dataset.drop(columns = ['Profit'])
X = pd.get_dummies(dataset)
X = X.values
X = X[:,0:5]
#getting the dummy columns for the categorical values
'''# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]
'''
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# Fitting Multiple Linear Regression to the Training set
#the number of independent variable is more than one thus creating a one-many relation
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Assingmnet: Draw the graphs of the columns individualy
X = dataset.iloc[:,0:1].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=0)

from sklearn.linear_model import LinearRegression
obj = LinearRegression()
obj.fit(X_train,y_train)

y_pred = obj.predict(X_test) 

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, obj.predict(X_train), color = 'blue')
plt.title('R&D spend vs Profit(Training Set)')
plt.xlabel('R & D Spend')
plt.ylabel('Profit')
plt.show()

X2 = dataset.iloc[:,1:2].values

from sklearn.model_selection import train_test_split
Ad_train,Ad_test,y_train,y_test = train_test_split(X2,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
obj2 = LinearRegression()
obj2.fit(Ad_train,y_train)

y_pred2 = obj2.predict(Ad_test)


plt.scatter(Ad_train, y_train, color = 'red')
plt.plot(Ad_train, obj2.predict(Ad_train), color = 'blue')
plt.title('Adminstration spend vs Profit(Training Set)')
plt.xlabel('Adminstration Spend')
plt.ylabel('Profit')
plt.show()









