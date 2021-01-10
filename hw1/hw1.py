#!/usr/bin/python
import mnist_reader as mr
import numpy as np
X_train, y_train = mr.load_mnist('', kind='train')
X_test, y_test = mr.load_mnist('', kind='t10k')
y_train1 = np.array([], dtype=np.int8)

weight = np.zeros(len(X_train[0]))
count = 0

#y_train1 = np.insert(y_train1, 0, 1)
#print(y_train1)

def learnRate(oldWeight, Xt, Yt):
    #a = np.linalg.norm(Xt)
    #b = (1-Yt*oldWeight*Xt)
    return (1-Yt*(oldWeight * Xt)) / (np.linalg.norm(Xt)**2)
    #return  b / a


def updatWeight(oldWeight, Xt, Yt):
    a = learnRate(oldWeight, Xt, Yt)

    newWeight = oldWeight + a * (Xt * Yt)
    #newWeight = np.add(a)
    return newWeight

for i in y_train:
    if i % 2 == 0:
        y_train1 = np.insert(y_train1, count, -1)
    else:
        y_train1 = np.insert(y_train1, count, 1)
    count += 1

for j in range(len(X_train)):
    if np.sign(np.inner(X_train[j], weight)) != y_train[j]:
        weight = updatWeight(weight, X_train[j], y_train[j])
print(weight)
#print(learnRate(weight, X_train[0], y_train[0]))
