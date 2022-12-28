# sigmoid with derivative in back prop

import numpy as np
import pandas as pd
import time
import random
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


class neural_network(object):
    def __init__(self):
        self.inputSize = 784
        self.outputSize = 10
        self.hiddenLayerSize = 64

        # random initialization
        self.w1 = np.random.randn(self.inputSize, self.hiddenLayerSize)
        self.w2 = np.random.randn(self.hiddenLayerSize, self.outputSize)
        self.b1 = np.random.randn(1, self.hiddenLayerSize)
        self.b2 = np.random.randn(1, self.outputSize)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return x > 0

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_prime(self, z):
        return np.exp(-z) / ((1 + np.exp(-z)) ** 2)

    def softmax_prime(self, z):
        f = np.exp(z)
        g = np.sum(f)
        print(f)
        print(g)
        return f / g - g ** (-2) * (f ** 2)

    def cost(self, a1, y):
        # s = 0
        f = self.forward(a1)
        rslt = f[3]
        s = (rslt - y) ** 2
        return np.sum(s) / (self.outputSize * 2)

    def forward(self, x):
        z2 = np.dot(x, self.w1) + self.b1
        a2 = self.relu(z2)
        z3 = np.dot(a2, self.w2) + self.b2
        # a3 = self.relu(z3)  # yHat
        yhat = self.softmax(z3)
        # print('yhat dim ', yhat.shape)

        return z2, a2, z3, yhat

    def back_prop(self, x, y):
        z2, a2, z3, a3 = self.forward(x)
        # print('dim a3', a3.shape)
        dc_dz3 = np.array((a3 - y) / self.outputSize)
        # print('dc_da3 dim ', dc_da3.shape)
        # da3_dz3 = self.sigmoid_prime(a3)
        # print('da3_dz3 dim ', da3_dz3.shape)
        # dc_dz3 = np.multiply(dc_da3, da3_dz3)
        # print('dc_dz3 dim ', dc_dz3.shape)
        dz3_dw2 = np.array(a2)  # (1, 30)
        # print('dz3_dw2 dim ', dz3_dw2.shape)
        dc_dw2 = np.dot(dz3_dw2.T, dc_dz3)
        # print('dc_dzw2 dim ', dc_dw2.shape, '\n')

        dz3_da2 = self.w2  # (30, 10)
        # print('dz3_da2 dim ', dz3_da2.shape)
        dc_da2 = np.dot(dz3_da2, dc_dz3.T)  # (30, 1)
        # print('dc_da2 dim ', dc_da2.shape)
        da2_dz2 = self.relu_prime(a2)
        # print('da2_dz2 dim ', da2_dz2.shape)
        dc_dz2 = np.multiply(dc_da2.T, da2_dz2)  # (30, 1)
        # print('dc_dz2 shape:', dc_dz2.shape)
        dz2_dw1 = np.array([x])  # (1, 784)
        # print('dim dz2_dw1: ', dz2_dw1.shape)
        dc_dw1 = np.dot(dz2_dw1.T, dc_dz2)
        # print('dc_dz3 dim ', dc_dz3.shape, '\n')

        # dc_db2 = np.dot(dc_da3, da3_dz3)

        return dc_dw1, dc_dw2, dc_dz2, dc_dz3

    def prediction(self, x):
        f = self.forward(x)
        a3 = f[3]
        # print(a3)
        return np.argmax(a3, 1)

    def update(self, w1, w2, dw1, dw2, lr):
        w1 = w1 - lr * dw1
        w2 = w2 - lr * dw2
        return w1, w2

    def accuracy(self, x, y):
        prdt = self.prediction(x)
        t = 0
        for i in range(len(y)):
            if prdt[i] == y[i]:
                t += 1
        return t / y.size



start = time.time()
nn = neural_network()
# print('layer 1 weights:', '\n', nn.w1)
# print('layer 2 weights:', '\n', nn.w2)

data = pd.read_csv(r"C:\Users\ASUS\Downloads\mnist_train_small.csv")
train, test = train_test_split(data, test_size=0.5, random_state=0)
train = np.array(train).T
test = np.array(test).T

train_label = train[0]
x_train = train[1:].T
x_train = x_train / 255
test_label = test[0]
x_test = test[1:].T
x_test = x_test / 255

y_train_set = []
for i in range(len(train_label)):
    temp = []
    for j in range(10):
        if train_label[i] == j:
            temp.append(1)
        else:
            temp.append(0)
    y_train_set.append(temp)

y_train_set = np.array(y_train_set)

batches = np.array([x_train[:100]])
label_batches = np.array([train_label[:100]])
y_batches = np.array([y_train_set[:100]])
for i in range(1, 99):
    a = np.array([x_train[i * 100:(i + 1) * 100]])
    b = np.array([train_label[i * 100:(i + 1) * 100]])
    c = np.array([y_train_set[i * 100:(i + 1) * 100]])
    batches = np.concatenate((batches, a))
    label_batches = np.concatenate((label_batches, b))
    y_batches = np.concatenate((y_batches, c))

# print(nn.back_prop(batches[0][0], y_batches[0][0])[0].shape)
print("check 1: ", time.time()-start)
c2 = time.time()
iters = 2000  # number pf iterations
learn_rate = 0.1  # learning rate
# plotting the accuracy
x_plot = []
y_plot = []
for i in range(iters):
    n = random.randint(0, 98)
    # derivative with respect to weights of the first layer
    dw1 = np.zeros((nn.inputSize, nn.hiddenLayerSize))
    # derivative with respect to weights of the second layer
    dw2 = np.zeros((nn.hiddenLayerSize, nn.outputSize))
    db1 = np.array([np.zeros(nn.hiddenLayerSize)])
    db2 = np.array([np.zeros(nn.outputSize)])
    for j in range(100):
        dw1 += nn.back_prop(batches[n][j], y_batches[n][j])[0]
        dw2 += nn.back_prop(batches[n][j], y_batches[n][j])[1]
        db1 += nn.back_prop(batches[n][j], y_batches[n][j])[2]
        db2 += nn.back_prop(batches[n][j], y_batches[n][j])[3]
    nn.w1 = nn.w1 - dw1 * learn_rate
    nn.w2 = nn.w2 - dw2 * learn_rate
    nn.b1 = nn.b1 - db1 * learn_rate
    nn.b2 = nn.b2 - db2 * learn_rate
    if i % 10 == 0:
        print('iter', i)
        print('train set acuuracy', nn.accuracy(x_train, train_label))
        print('test accuracy: ', nn.accuracy(x_test, test_label))
    x_plot.append(i)
    y_plot.append(nn.accuracy(x_train, train_label) * 100)

print('test set accuracy', nn.accuracy(x_test, test_label))

# img = np.array([x_train[5000][:28]])
# for i in range(1, 28):
#     a = np.array([x_train[5000][i * 28: i * 28 + 28]])
#     # print(a)
#     img = np.concatenate((img, a))
# print(img.shape)
# # imgplt = plt.imshow(img)
# print('prediction:', nn.prediction(x_train)[5000])

# plt.show(block=False)
end = time.time()
plt.scatter(x_plot, y_plot)
plt.xlabel('Iteration')
plt.ylabel('Train set acuuracy')
# plt.title(label='Final train set accuracy: ' + str(int(round(nn.accuracy(x_train, train_label), 2) * 100)) + '%' + '\n'+'Final test set accuracy: '+ str(int(round(nn.accuracy(x_test, test_label), 2) * 100)) + '%')
print('Final train set accuracy:', str(int(round(nn.accuracy(x_train, train_label), 2) * 100)) + '%')
print('Final test set accuracy:', str(int(round(nn.accuracy(x_test, test_label), 2) * 100)) + '%')
print('Runtime:', round(end - c2, 2), 'seconds')
plt.show()
