#!/usr/bin/python
import mnist_reader as mr
import numpy as np
import matplotlib.pyplot as plt
X_train, y_train = mr.load_mnist('', kind='train')
X_test, y_test = mr.load_mnist('', kind='t10k')

weight = np.zeros([len(X_train[0]) * 10])

def functionX(Xt):
    newX = np.zeros([10, len(Xt) * 10]) #len(Xt) = 784
    for i in range(10):
        for j in range(len(Xt)):
            newX[i][i*len(Xt)+j] = Xt[j]
    return newX
#modify
def learnRate(oldWeight, Xt, Yt, y):
    return (1-(np.dot(oldWeight, Xt[Yt]) - np.dot(oldWeight, Xt[y]))) / (np.linalg.norm(np.subtract(Xt[Yt], Xt[y]))**2)

def updatWeight(oldWeight, Xt, y, Yt):
    newWeight = oldWeight + np.dot(learnRate(oldWeight, Xt, Yt, y), np.subtract(Xt[Yt], Xt[y]))
    return newWeight

def timeWeight(Xt, w):
    innerResult = np.array([], dtype=np.float)
    for i in range(len(Xt)): #len(Xt) = 10
        innerResult = np.insert(innerResult, i, np.inner(Xt[i], w))
    return innerResult

X_result = np.array([], dtype=np.float)
for i in range(50):
    countMistake = 0
    for j in range(len(X_train)):
        newX = functionX(X_train[j])
        innerResult = timeWeight(newX, weight)
        y = np.argmax(innerResult)
        #mistake
        if y != y_train[j]:
            countMistake += 1
            weight = updatWeight(weight, newX, y, y_train[j])
    print(i+1, '/50')
    X_result = np.insert(X_result, i, countMistake)
#print(X_result)
plt.title("PA_multi(a)")
plt.xlabel('training iterations')
plt.ylabel('mistakes')
plt.plot([i for i in range(50)], X_result)
plt.show()
