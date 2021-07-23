from math import gamma
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
print(array[6493])
print(array.shape)
array = np.delete(array, np.where(np.isin("nan", array)),axis=1)
print(array.shape)
print(array[6493])
# X = array[:,1:10]
# y = array[:,11]
# X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# model = SVC(gamma='auto')
# model.fit(X_train, Y_train)