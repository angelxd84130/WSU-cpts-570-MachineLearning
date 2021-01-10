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
y_test1 = binaryY(y_test)
X_train_result = np.array([], dtype=np.float)
X_test_result = np.array([], dtype=np.float)

for i in range(20):
    count_train = 0
    count_test = 0
    #train
    for j in range(len(X_train)):
        if np.sign(np.inner(X_train[j], weight)) != y_train1[j]:
            weight = updatWeight(weight, X_train[j], y_train1[j])
        else:
            count_train += 1
    X_train_result = np.insert(X_train_result, i, count_train / len(X_train))
    #test
    for j in range(len(X_test)):
        if np.sign(np.inner(X_test[j], weight)) == y_test1[j]:
            count_test += 1
    X_test_result = np.insert(X_test_result, i, count_test / len(X_test))
plt.title("Perceptron_binary(b)")
plt.xlabel('training iterations')
plt.ylabel('accuracy')
plt.plot([i for i in range(20)], X_train_result, label="train")
plt.plot([i for i in range(20)], X_test_result, label="test")
plt.legend()
plt.show()
