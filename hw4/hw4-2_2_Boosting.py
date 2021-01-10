#!/usr/bin/python
import mnist_reader as mr
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
X_train, y_train = mr.load_mnist('', kind='train')
X_test, y_test = mr.load_mnist('', kind='t10k')

X_train = X_train[:2000]
y_train = y_train[:2000]

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
#print(clf.predict(y_test))
print(accuracy_score(y_test, clf.predict(X_test)))

import torch
x = torch.linspace(0, 1)