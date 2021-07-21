import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LogisticRegression

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
names = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
dataset = pd.read_csv(url)
dataset = dataset.iloc[1:, :]
#print(dataset.head())
array = dataset.values
X = array[:, 0:10]
y = array[11,11]
model = LogisticRegression()
model.fit(X, y)