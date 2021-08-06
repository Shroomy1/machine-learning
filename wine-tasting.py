from math import gamma, isnan
from operator import ne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.construct import rand
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
import warnings
warnings.filterwarnings('ignore')
data = "winequalityN.csv"
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
names = ["fixed acidity","volatile acidity","citric acid","residual sugar","chlorides","free sulfur dioxide","total sulfur dioxide","density","pH","sulphates","alcohol","quality"]
df = pd.read_csv(url, sep=";", header=0)
print(df.head())
array = df.values
print(array)
print(array[1500])
print(array.shape)
#new_array = np.delete(array, np.where(np.isin(array, "")), axis=1)
new_array = df[(df.iloc[:, 1:] != 0).all(1)]
new_array = new_array.values
print("DELETED ROWS")
print(new_array.shape)
# print(new_array[6493])
X = new_array[:,0:11]
y = new_array[:,11]
print("SET X AND Y")
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=42)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
models.append(('Random_Forest',RandomForestClassifier(n_estimators=10)))
models.append(('Bagging', BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)))
models.append(('SGD',SGDClassifier(loss="hinge", penalty="l2", max_iter=5)))

results = []
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

plt.boxplot(results, labels=names)
plt.title('Algorithm Comparison')
# df.plot(kind='box', subplots=True, layout=(12,1), sharex=False, sharey=False)
# df.plot.scatter(x=X, y=y)
plt.show()

model = DecisionTreeClassifier()
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)
print("ACCURACCY SCORE")
print(accuracy_score(Y_validation, predictions))
print("CONFUSION MATRIX")
print(confusion_matrix(Y_validation, predictions))
sn.heatmap(confusion_matrix(Y_validation, predictions), annot=True)
print("CLASSIFICATION REPORT")
print(classification_report(Y_validation, predictions, zero_division=1))
plt.show()

print("-------------------------------------------")
print("MAKING PREDICTIONS")
# input_array = np.array['7' '0.27' '0.36' '20.7' '0.045' '45' '170' '1.001' '3' '0.45' '8.8']
input_array = []
input_array_size = int(input("Array Size: "))

for i in range(input_array_size):
	input_array_value_input = input("Element: ")
	input_array = np.append(input_array, input_array_value_input)
print(input_array)

input_array = np.reshape(input_array,(1,input_array.size))

prediction = model.predict(input_array)
print(prediction)