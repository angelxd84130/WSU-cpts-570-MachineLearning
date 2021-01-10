#!/usr/bin/python
from sklearn.ensemble import BaggingClassifier
import mnist_reader as mr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
X_train, y_train = mr.load_mnist('', kind='train')
X_test, y_test = mr.load_mnist('', kind='t10k')


result = np.array([], dtype=float)

BagModel = BaggingClassifier()


BagModel.fit(X_train, y_train)
result = np.insert(result, i, accuracy_score(y_test, BagModel.predict(X_test)))
plt.title("Bagging")
plt.xlabel("iteration")
plt.ylabel("accuracy")
plt.plot([i for i in range(5)], result)
plt.show()

