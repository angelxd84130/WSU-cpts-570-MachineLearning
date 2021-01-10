#!/usr/bin/python
import time
import mnist_reader as mr
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

X_train, y_train = mr.load_mnist('data/', kind='train')
X_test, y_test = mr.load_mnist('data/', kind='t10k')

X_valid = X_train[48000:]
y_valid = y_train[48000:]
X_train1 = X_train[:48000]
y_train1 = y_train[:48000]

train_result = np.array([], dtype=np.float)
valid_result = np.array([], dtype=np.float)
test_result = np.array([], dtype=np.float)

C = 0.0001
print("Start!", time.strftime("%H:%M:%S"))
for i in range(4):
    print("start fitting", i, "/ 4")
    if i == 0:
        svm_model = svm.LinearSVC(C=C)
    else:
        svm_model = svm.SVC(C=C, kernel='poly', degree=i+1)
    svm_model.fit(X_train1, y_train1)
    print("finish fitting", i, "/ 4!")
    train_result = np.insert(train_result, i, accuracy_score(svm_model.predict(X_train1), y_train1))
    valid_result = np.insert(valid_result, i, accuracy_score(svm_model.predict(X_valid), y_valid))
    test_result = np.insert(test_result, i, accuracy_score(svm_model.predict(X_test), y_test))
    print("finish", i, "/ 4", time.strftime("%H:%M:%S"))


plt.title("SVM_c")
plt.plot(['linear', 'poly_kernel_2', 'poly_kernel_3', 'poly_kernel_4'], train_result, label='train')
plt.plot(['linear', 'poly_kernel_2', 'poly_kernel_3', 'poly_kernel_4'], valid_result, label='valid')
plt.plot(['linear', 'poly_kernel_2', 'poly_kernel_3', 'poly_kernel_4'], test_result, label='test')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.legend()
plt.show()