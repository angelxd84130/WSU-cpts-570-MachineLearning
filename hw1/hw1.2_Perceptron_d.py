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

def updatWeight(oldWeight, Xt, y, Yt):
    newWeight = oldWeight + np.subtract(Xt[Yt], Xt[y])
    return newWeight

def timeWeight(Xt, w):
    innerResult = np.array([], dtype=np.float)
    for i in range(len(Xt)): #len(Xt) = 10
        innerResult = np.insert(innerResult, i, np.inner(Xt[i], w))
    return innerResult

Xresult = np.array([], dtype=np.float)
count = 0
for i in range(20):
    #train
    for j in range(len(X_train)):
        newX = functionX(X_train[j])
        innerResult = timeWeight(newX, weight)
        y = np.argmax(innerResult)
        #mistake
        if y != y_train[j]:
            weight = updatWeight(weight, newX, y, y_train[j])

        # test
        if j > 0 and j % 4999 == 0:
            count_test = 0
            for k in range(len(X_test)):
                newX = functionX(X_test[k])
                innerResult = timeWeight(newX, weight)
                y = np.argmax(innerResult)
                if y == y_test[k]:
                    count_test += 1
            # print("countTest: ", count_test)
            Xresult = np.insert(Xresult, count, count_test / len(X_test))
            count += 1

    print(i+1, '/20')

#print(X_result)
plt.title("Perceptron_multi(d)")
plt.xlabel('training iterations')
plt.ylabel('accuracy')
plt.plot([i for i in range(5000, 1200001, 5000)], Xresult)

plt.show()
