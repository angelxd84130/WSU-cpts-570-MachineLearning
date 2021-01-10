#!/usr/bin/python
import mnist_reader as mr
import numpy as np
import matplotlib.pyplot as plt
X_train, y_train = mr.load_mnist('', kind='train')
X_test, y_test = mr.load_mnist('', kind='t10k')

weight = np.zeros([len(X_train[0]) * 10])
uWeight = np.zeros([len(X_train[0]) * 10])

def functionX(Xt):
    newX = np.zeros([10, len(Xt) * 10]) #len(Xt) = 784
    for i in range(10):
        for j in range(len(Xt)):
            newX[i][i*len(Xt)+j] = Xt[j]
    return newX

def updatWeight(oldWeight, Xt, y, Yt):
    newWeight = oldWeight + np.subtract(Xt[Yt], Xt[y])
    return newWeight

def timeWeight(Xt, w):
    innerResult = np.array([], dtype=np.float)
    for i in range(len(Xt)): #len(Xt) = 10
        innerResult = np.insert(innerResult, i, np.inner(Xt[i], w))
    return innerResult

count = 1
for i in range(20):

    #train
    for j in range(len(X_train)):
        newX = functionX(X_train[j])
        innerResult = timeWeight(newX, weight)
        y = np.argmax(innerResult)
        #mistake
        if y != y_train[j]:
            weight = updatWeight(weight, newX, y, y_train[j])
            uWeight = uWeight + np.dot(np.dot(count, y_train[j]), newX)
        count += 1
    print(i+1, '/20')
uWeight = weight - uWeight / count

#test
count_Ava = 0
for j in range(len(X_test)):
    newX = functionX(X_test[j])
    innerResult = timeWeight(newX, uWeight)
    y = np.argmax(innerResult)
    if y == y_test[j]:
        count_Ava += 1
print("Average accuracy: ", count_Ava / len(X_test))

count_Plain = 0
for j in range(len(X_test)):
    newX = functionX(X_test[j])
    innerResult = timeWeight(newX, weight)
    y = np.argmax(innerResult)
    if y == y_test[j]:
        count_Plain += 1
print("Plain: accuracy", count_Plain / len(X_test))
