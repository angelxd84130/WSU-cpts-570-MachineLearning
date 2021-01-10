#!/usr/bin/python
import mnist_reader as mr
import numpy as np

X_train, y_train = mr.load_mnist('', kind='train')
X_test, y_test = mr.load_mnist('', kind='t10k')

weight = np.zeros(len(X_train[0]))
uWeight = np.zeros((len(X_train[0])))

def updatWeight(oldWeight, Xt, Yt):
    newWeight = oldWeight + np.dot(Xt, Yt)
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

count = 1
for i in range(20):
    #train
    for j in range(len(X_train)):
        if np.sign(np.dot(X_train[j], weight)) != y_train1[j]:
            weight = updatWeight(weight, X_train[j], y_train1[j])
            uWeight = uWeight + np.dot(np.dot(count, y_train1[j]), X_train[j])
        count += 1

uWeight = weight - uWeight / count
#test
countAva = 0
for i in range(len(X_test)):
    if np.sign(np.dot(X_test[i], uWeight)) == y_test1[i]:
        countAva += 1
print("Average accuracy: ", countAva / len(X_test))
#0.9455

countPlain = 0
for i in range(len(X_test)):
    if np.sign(np.dot(X_test[i], weight)) == y_test1[i]:
        countPlain += 1
print("Plain: accuracy", countPlain / len(X_test))
#0.9459
