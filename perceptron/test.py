import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
import matplotlib.pyplot as plt
from pandas.core.algorithms import mode
import per as perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random

random.seed(40)


data = pd.read_csv("data.csv", names=["a","b","c"])
X = data[["a","b"]].values
y = data[["c"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
model = perceptron.perceptron()
model.fit(X_train,y_train)
predictions = model.predict(X_test)

acc = accuracy_score(y_test, predictions)

print (acc)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.scatter(X_train[:, 0], X_train[:, 1], marker="o", c=y_train)

x0_1 = np.amin(X_train[:, 0])
x0_2 = np.amax(X_train[:, 0])

x1_1 = (-model.w[0] * x0_1 - model.b) / model.w[1]
x1_2 = (-model.w[0] * x0_2 - model.b) / model.w[1]

ax.plot([x0_1, x0_2], [x1_1, x1_2], "r")

ymin = np.amin(X_train[:, 1])
ymax = np.amax(X_train[:, 1])

ax.set_ylim([ymin - 3, ymax + 3])
plt.show()