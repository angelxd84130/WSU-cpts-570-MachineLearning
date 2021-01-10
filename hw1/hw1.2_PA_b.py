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

X_train_result = np.array([], dtype=np.float)
X_test_result = np.array([], dtype=np.float)
for i in range(20):
    count_train = 0
    count_test = 0
    #train
    for j in range(len(X_train)):
        newX = functionX(X_train[j])
        innerResult = timeWeight(newX, weight)
        y = np.argmax(innerResult)
        #mistake
        if y != y_train[j]:
            weight = updatWeight(weight, newX, y, y_train[j])
        else:
            count_train += 1
    X_train_result = np.insert(X_train_result, i, count_train / len(X_train))
    for j in range(len(X_test)):
        newX = functionX(X_test[j])
        innerResult = timeWeight(newX, weight)
        y = np.argmax(innerResult)
        if y == y_test[j]:
            count_test += 1
    X_test_result = np.insert(X_test_result, i, count_test / len(X_test))
    print(i+1, '/20')

#print(X_result)
plt.title("PA_multi(b)")
plt.xlabel('training iterations')
plt.ylabel('accuracy')
plt.plot([i for i in range(20)], X_train_result, label = 'train')
plt.plot([i for i in range(20)], X_test_result, label = 'test')
plt.legend()
plt.show()
print("Xtrain: ", X_train_result)
print("Xtest: ", X_test_result)