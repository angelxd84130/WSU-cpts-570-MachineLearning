#!/usr/bin/python
import mnist_reader as mr
import numpy as np
X_train, y_train = mr.load_mnist('', kind='train')
X_test, y_test = mr.load_mnist('', kind='t10k')
y_train1 = np.array([], dtype=np.int8)

weight = np.zeros(len(X_train[0]))
count = 0

def updatWeight(oldWeight, Xt, Yt):
    newWeight = oldWeight + (Xt * Yt)
    return newWeight

def binaryY(Y):
    count = 0
    Y1 = np.array([], dtype=np.int8)
    for i in Y:
        if i % 2 == 0:
            Y1 = np.insert(Y1, count, -1)
        else:
            Y1 = np.insert(Y1, count, 1)
        count += 1
    return Y1
y_train1 = binaryY(y_train)

for j in range(len(X_train)):
    if np.sign(np.inner(X_train[j], weight)) != y_train1[j]:
        weight = updatWeight(weight, X_train[j], y_train1[j])
print(weight)
#print(learnRate(weight, X_train[0], y_train[0]))
