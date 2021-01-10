#!/usr/bin/python
import mnist_reader as mr
import numpy as np
import matplotlib.pyplot as plt
X_train, y_train = mr.load_mnist('', kind='train')
X_test, y_test = mr.load_mnist('', kind='t10k')


weight = np.zeros(len(X_train[0]))

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
X_result = np.array([], dtype=np.int)

for i in range(50):
    countMistake = 0
    for j in range(len(X_train)):
        if np.sign(np.inner(X_train[j], weight)) != y_train1[j]:
            countMistake += 1
            weight = updatWeight(weight, X_train[j], y_train1[j])

    X_result = np.insert(X_result, i, countMistake)
print(X_result)
plt.title("Perceptron_binary(a)")
plt.xlabel('training iterations')
plt.ylabel('mistakes')
plt.plot([i for i in range(50)], X_result)
plt.show()
