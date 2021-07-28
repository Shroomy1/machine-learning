from math import gamma, isnan
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

data = "winequalityN.csv"
names = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
df = pd.read_csv(data)
# print(df.head())
array = df.values
fixed_data = "fixed_wine.csv"
df2 = pd.read_csv(fixed_data)
fixed_array = df2.values
print(array[6493])
print(array.shape)
#new_array = np.delete(array, np.where("nan" in array), axis=1)
print(fixed_array.shape)
print(fixed_array[6465])
# X = array[:,1:10]
# y = array[:,11]
# X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# model = SVC(gamma='auto')
# model.fit(X_train, Y_train)