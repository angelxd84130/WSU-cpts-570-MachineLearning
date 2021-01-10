#!/usr/bin/python
import time
import mnist_reader as mr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

X_train, y_train = mr.load_mnist('data/', kind='train')
X_test, y_test = mr.load_mnist('data/', kind='t10k')

X_valid = X_train[1000:1200]
y_valid = y_train[1000:1200]
X_train1 = X_train[:1000]
y_train1 = y_train[:1000]
gama = np.zeros(len(X_train1))
y_pre = np.zeros(len(X_valid))
b = 0


def phiX(Xt):
    newX = np.zeros([10, len(Xt) * 10]) #len(Xt) = 784
    for i in range(10):
        for j in range(len(Xt)):
            newX[i][i*len(Xt)+j] = Xt[j]
    return newX


def k_poly(xi, xj):
    return (np.inner(xi, xj) + 1)**2
    #return np.inner(xi, xj)


def k_matrix():
    k = np.zeros([len(X_train1), len(X_train1)])
    for i in range(len(X_train1)):
        for j in range(len(X_train1)):
            k[i][j] = k_poly(i, j)
    return k


print("Start fitting", time.strftime("%H:%M:%S"))
k = k_matrix()


for n in range(len(X_train1)):
    #f_x = phiX(np.zeros(len(X_train1[0])))
    f_x = 0
    for m in range(0, n+1):
        f_x += np.dot(gama[m], k[m][n]) + b
    if y_train[n] * f_x <= 0:
        gama[n] = gama[n] + y_train[n]
        b = b + y_train[n]
    #print("fitting", n+1, "/100: f_x:", f_x)


print("Finish fitting! Start predict:")
for i in range(len(X_valid)):
    f_x_1 = phiX(np.zeros(len(X_valid[0])))

    for j in range(i+1):
        f_x_1 += gama[j] * phiX(X_valid[j])
    y_pre[i] = np.argmin(np.linalg.norm(phiX(X_valid[i]) - f_x_1)**2)
    #print("predicting", i+1, "/20")

print(y_valid)
print(y_pre)
