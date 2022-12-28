import pandas as pd
import numpy as np
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


class neural_network(object):
    def __init__(self):
        self.inputSize = 784
        self.outputSize = 10
        self.hiddenLayerSize = 64

        # random initialization
        self.w1 = loadtxt('layer1_weights.csv')
        self.w2 = loadtxt('layer2_weights.csv')

    def relu(self, x):
        return np.maximum(0, x)

    def relu_prime(self, x):
        return x > 0

    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))

    def softmax_prime(self, z):
        f = np.exp(z)
        g = np.sum(f)
        return f / g - g * f ** 2

    def cost(self, a1, y):
        # s = 0
        f = self.forward(a1)
        rslt = f[3]
        s = (rslt - y) ** 2
        return np.sum(s) / (self.outputSize * 2)

    def forward(self, x):
        z2 = np.dot(x, self.w1)
        a2 = self.relu(z2)
        z3 = np.dot(a2, self.w2)
        a3 = self.relu(z3)  # yHat
        yhat = self.softmax(a3)

        return z2, a2, z3, yhat

    def back_prop(self, x, y):
        z2, a2, z3, a3 = self.forward(x)
        dc_da3 = (a3 - y) / self.outputSize
        da3_dz3 = self.relu_prime(a3)
        dc_dz3 = np.array([np.multiply(dc_da3, da3_dz3)])
        dz3_dw2 = np.array([a2])
        dc_dw2 = np.dot(dz3_dw2.T, dc_dz3)

        dz3_da2 = self.w2
        dc_da2 = np.dot(dz3_da2, dc_dz3.T)
        da2_dz2 = self.relu_prime(a2)
        dc_dz2 = np.multiply(dc_da2.T, da2_dz2)
        dz2_dw1 = np.array([x])
        dc_dw1 = np.dot(dz2_dw1.T, dc_dz2)

        return dc_dw1, dc_dw2

    def prediction(self, x):
        f = self.forward(x)
        a3 = f[3]
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


def get_image(pxls):
    img = np.array([pxls[:28]])
    for i in range(1, 28):
        a = np.array([pxls[i * 28: i * 28 + 28]])
        img = np.concatenate((img, a))
    return img


nn = neural_network()


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

imgs = []
n = int(input('enter a digit: '))
i = 0
np.random.shuffle(x_train)
for j in range(4):
    b = True
    while b:
        if nn.prediction(x_train)[i] == n:
            imgs.append(get_image(x_train[i]))
            b = False
        i += 1

fig = plt.figure()

rows = 2
columns = 2
fig.add_subplot(rows, columns, 1)
plt.imshow(imgs[0], cmap='gray')
plt.axis('off')
fig.add_subplot(rows, columns, 2)
plt.imshow(imgs[1], cmap='gray')
plt.axis('off')
fig.add_subplot(rows, columns, 3)
plt.imshow(imgs[2], cmap='gray')
plt.axis('off')
fig.add_subplot(rows, columns, 4)
plt.imshow(imgs[3], cmap='gray')
plt.axis('off')
fig.suptitle('Sample images of ' + str(n), fontsize=16)
plt.show()
