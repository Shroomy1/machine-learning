import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.linear_model import LinearRegression


url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv"
names = ['Pregnancies', 'Glucose', 'BloodPressure', 'Skinfold', 'Insulin', 'BMI', 'Pedigree', 'Age', 'Outcome']
dataset = pd.read_csv(url, names= names)
# print(dataset.groupby("Outcome").size())
# print(dataset.describe())
#dataset.hist()
# corr_matrix = dataset.corr()
# sn.heatmap(corr_matrix, annot=True)
# plt.show()

model = LinearRegression(normalize=True)
print(model.normalize)
print(model)