#!/usr/bin/python
import time
import mnist_reader as mr
import matplotlib.pyplot as plt
import numpy as np
import gc
from sklearn.metrics import accuracy_score

X_train, y_train = mr.load_mnist('data/', kind='train')
X_test, y_test = mr.load_mnist('data/', kind='t10k')
#Normelize
X_train = X_train/255
X_test = X_test/255

X_valid = X_train[4800:6000]
y_valid = y_train[4800:6000]
X_train1 = X_train[:4800]
y_train1 = y_train[:4800]
X_test1 = X_test[:1000]
y_test1 = y_test[:1000]
gama = np.zeros([len(X_train1), 10])

'''
def k_poly(xi, xj):
    return (np.dot(xi, xj) + 1)**2


def k_matrix(Xt, yt):
    k = np.zeros([len(Xt), len(yt)])
    print("Xt:", len(Xt), "yt", len(yt))
    for i in range(len(Xt)):
        for j in range(len(yt)):
            k[i][j] = k_poly(Xt[i], yt[j])
        print("i:", i, "j:", j, "k:", k[i][j])
    print("finish k")
    return k
'''
'''
print("Normalizing..", time.strftime("%H:%M:%S"))
X_training = np.zeros([len(X_train1), len(X_train1[0])])
for i in range(len(X_train1)):
    for j in range(len(X_train1[0])):
        X_training[i][j] = (X_train1[i][j] - np.mean(X_train1[i])) / np.std(X_train1[i])
del X_train1
gc.collect()
print("Finish training")
X_validation = np.zeros([len(X_valid), len(X_valid[0])])
for i in range(len(X_valid)):
    for j in range(len(X_valid[0])):
        X_validation[i][j] = (X_valid[i][j] - np.mean(X_valid[i])) / np.std(X_valid[i])
del X_valid
gc.collect()
print("Finish validation")
X_testing = np.zeros([len(X_test), len(X_test)])
for i in range(len(X_test)):
    for j in range(len(X_test[0])):
        X_testing[i][j] = (X_test[i][j] - np.mean(X_test)) / np.std(X_test)
del X_test
gc.collect()
print("Finish testing")
'''
'''
print("Build k matrix", time.strftime("%H:%M:%S"))
k_train = k_matrix(X_train1, X_train1)
print("get k_train")
k_valid = k_matrix(X_train1, X_valid)
k_test = k_matrix(X_train1, X_test)
'''
train_result = np.array([], dtype=np.float)
valid_result = np.array([], dtype=np.float)
test_result = np.array([], dtype=np.float)


for t in range(5):
    print("Start fitting..", t+1, "/5", time.strftime("%H:%M:%S"))
    mistake = correct = 0
    for n in range(len(X_train1)):
        #run each example
        #W1 = a11 * k(x1, xn) + a21 * k(x2, xn)
        weight = np.zeros(10)
        for j in range(10):
            for i in range(len(X_train1)):
                weight[j] += np.inner(gama[i][j], (np.inner(X_train1[i], X_train1[n])+1)**2)
        y_pre = np.argmax(weight)
        if y_pre != y_train1[n]:
            gama[n][y_train1[n]] = 1
            gama[n][y_pre] = -1
            mistake += 1
        else:
            correct += 1
        if n % 99 == 0:
            print(n, "/", len(X_train1))
    train_result = np.insert(train_result, t, mistake*10)
    print("Finish fitting\ncorrect:", correct/n)

    print("Start Valid", time.strftime("%H:%M:%S"))
    mistake = correct = 0
    for m in range(len(X_valid)):
        weight = np.zeros(10)
        for j in range(10):
            for i in range(len(X_train1)):
                weight[j] += np.inner(gama[i][j], (np.inner(X_train1[i], X_valid[m])+1)**2)
        y_pre = np.argmax(weight)
        if y_pre != y_valid[m]:
            mistake += 1
        else:
            correct += 1
        if m % 99 == 0:
            print(m, "/", len(X_valid))
    valid_result = np.insert(valid_result, t, mistake*10)
    print("Finish validation\ncorrect:", correct/m)

    print("Start Test", time.strftime("%H:%M:%S"))
    mistake = correct = 0
    for o in range(len(X_test1)):
        weight = np.zeros(10)
        for j in range(10):
            for i in range(len(X_train1)):
                weight[j] += np.inner(gama[i][j], (np.inner(X_train1[i], X_test1[o])+1)**2)
        y_pre = np.argmax(weight)
        if y_pre != y_test1[o]:
            mistake += 1
        else:
            correct += 1
        if o % 99 == 0:
            print(o, "/", len(X_test1))
    test_result = np.insert(test_result, t, mistake*10)
    print("Finish testing\ncorrect:", correct / o)
    gc.collect()

plt.title("Kernelized Perceptron")
plt.plot([i for i in range(1, 6)], train_result, label='train')
plt.plot([i for i in range(1, 6)], valid_result, label='valid')
plt.plot([i for i in range(1, 6)], test_result, label='test')
plt.legend()
plt.xlabel("iterations")
plt.ylabel("mistakes")
plt.show()
