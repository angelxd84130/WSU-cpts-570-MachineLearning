import matplotlib.pyplot as plt
from torch import tensor
import torch
import matplotlib as mpl
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
#import mnist_reader as mr
#X_train, y_train = mr.load_mnist('data/', kind='train')
#X_test, y_test = mr.load_mnist('data/', kind='t10k')
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv1D

model = Sequential()
model.add(Conv1D(filters=1, kernel_size=5, strides=2, padding=2, activation="relu"))
'''
def get_data():
    train_data = pd.read_csv('data/fashion-mnist_train.csv')
    test_data = pd.read_csv('data/fashion-mnist_test.csv')
    x_train = train_data[train_data.columns[1:]].values
    y_train = train_data.label.values
    x_test = test_data[test_data.columns[1:]].values
    y_test = test_data.label.values
    return map(tensor, (x_train, y_train, x_test, y_test))

x_train, y_train, x_test, y_test = get_data()
train_m, train_n = x_train.shape
test_m, test_n = x_test.shape
print(train_n, train_m, test_n, test_m)
n_cls = y_train.max()+1
mpl.rcParams['image.cmap'] = 'gray'
plt.imshow(x_train[torch.randint(train_m, (1,))].view(28, 28))

#creat model
class FashionMnistNet(nn.Module):
    # Based on Lecunn's Lenet architecture
    def __init__(self):
        super(FashionMnistNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3) # convolution demo (http://cs231n.github.io/convolutional-networks/)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

#instantiating the model
#model = FashionMnistNet()
#print(model)
#model.forward(X_train.float().reshape(train_f, 1, 28, 28))
model = FashionMnistNet() # Creating a model
lr = 0.05 # learning rate
epochs = 10 # number of epochs
bs = 32 # batch size
loss_func = F.cross_entropy # loss function
opt = optim.Adam(model.parameters(), lr=lr) # optimizer
accuracy_vals = []
for epoch in range(epochs):
    model.train()
    # print(model.training)
    for i in range((
                           train_n - 1) // bs + 1):  # (train_n-1)//bs equals the number of batches when we divide the divide by given batch size bs
        start_i = i * bs
        end_i = start_i + bs
        # Pytorch reshape function has four arguments -  (batchsize, number of channels, width, height)
        xb = x_train[start_i:end_i].float().reshape(bs, 1, 28, 28)
        yb = y_train[start_i:end_i]
        loss = loss_func(model.forward(xb), yb)  # model.forward(xb) computes the prediction of model on given input xb
        loss.backward()  # backpropagating the gradients
        opt.step()  # gradient descent
        opt.zero_grad()  # don't forget to add this line after each batch (zero out the gradients)

    model.eval()
    # print(model.training)
    with torch.no_grad():  # this line essentially tells pytorch don't compute the gradients for test case
        total_loss, accuracy = 0., 0.
        for i in range(test_n):
            x = x_test[i].float().reshape(1, 1, 28, 28)
            y = y_test[i]
            pred = model.forward(x)
            accuracy += (torch.argmax(pred) == y).float()
        print("Accuracy: ", (accuracy * 100 / test_n).item())
        accuracy_vals.append((accuracy * 100 / test_n).item())
plt.plot(accuracy_vals)
plt.show()