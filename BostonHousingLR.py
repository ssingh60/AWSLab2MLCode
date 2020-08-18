import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pickle


boston_dataset = pd.read_csv("BostonHousing.csv")

boston_dataset.head()
boston_dataset.isnull().sum()

X = boston_dataset[['lstat','rm']]
Y = boston_dataset[['medv']]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))

y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

Pkl_Filename = "Pickle_LR_Model.pkl"  


with open(Pkl_Filename, 'wb') as file:  
    pickle.dump(lin_model, file)