# Load data

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

x_train = np.load("x_train.npy")
t_train = np.load("y_train.npy")
x_test = np.load("x_test.npy")
t_test = np.load("y_test.npy")


# 1. Compute the mean vectors mi, (i=1,2) of each 2 classes

c = 2  # The number of class
x_train_0 = x_train[np.where(t_train == 0)]
x_train_1 = x_train[np.where(t_train == 1)]
m1 = np.mean(x_train_0, axis=0) * np.ones((1, x_train_0.shape[1]))
m2 = np.mean(x_train_1, axis=0) * np.ones((1, x_train_1.shape[1]))
print(f"mean vector of class 1: {m1}", f"mean vector of class 2: {m2}")


# 2. Compute the Within-class scatter matrix SW

Sw = np.dot((x_train_0 - np.multiply(m1, np.ones((x_train_0.shape)))).T,
            x_train_0 - np.multiply(m1, np.ones((x_train_0.shape)))) +\
     np.dot((x_train_1 - np.multiply(m2, np.ones((x_train_1.shape)))).T,
            x_train_1 - np.multiply(m2, np.ones((x_train_1.shape))))

assert Sw.shape == (2, 2)
print(f"Within-class scatter matrix SW: {Sw}")


# 3.  Compute the Between-class scatter matrix SB

SB = np.dot((m2 - m1).T, (m2 - m1))

assert SB.shape == (2, 2)
print(f"Between-class scatter matrix SB: {SB}")


# 4. Compute the Fisher’s linear discriminant

w, v = np.linalg.eig(np.dot(np.linalg.inv(Sw), SB))
weight = np.zeros((c, 1))
weight[:, 0] = v[:, np.argmax(w)]


assert weight.shape == (2, 1)
print(f" Fisher’s linear discriminant: {weight}")


# 5. Project the test data by linear discriminant to
#    get the class prediction by nearest-neighbor rule
#    and calculate the accuracy score

y_train = np.dot(x_train, weight)
y_test = np.dot(x_test, weight)

test_pred = np.zeros((x_test.shape[0], 1))
for i in range(x_test.shape[0]):
    test_pred[i, 0] = t_train[np.argmin(np.absolute(y_test[i] *
                                                    np.ones(y_train.shape) -
                                                    y_train))]

acc = accuracy_score(test_pred, t_test)
print(f"Accuracy of test-set = {acc}")


# 6. Plot

x_train_plot = np.ones((x_train.shape[0], 2))
x_train_plot[:, 0] = x_train[:, 0]
w_plot = np.dot(np.dot(np.linalg.inv(np.dot(x_train_plot.T, x_train_plot)),
                       x_train_plot.T), y_train)
x_plot = np.ones((100, 2))
x_plot[:, 0] = np.linspace(np.amin(x_train[:, 0]),
                           np.amax(x_train[:, 0]), 100)
y_plot = np.dot(x_plot, w_plot)

x_train_0_plot = np.ones((x_train_0.shape[0], 2))
x_train_0_plot[:, 0] = x_train_0[:, 0]
x_train_1_plot = np.ones((x_train_1.shape[0], 2))
x_train_1_plot[:, 0] = x_train_1[:, 0]

plt.figure()
plt.title(f'slope = {w_plot[0]}, intercept = {w_plot[1]}')

plt.plot(x_train_0[:, 0], x_train_0[:, 1], 'r.', label='class 1')
plt.plot(x_train_1[:, 0], x_train_1[:, 1], 'b.', label='class 2')

plt.plot(x_train_0[:, 0], np.dot(x_train_0_plot, w_plot),
         'm.', label='the projection of class 1')
plt.plot(x_train_1[:, 0], np.dot(x_train_1_plot, w_plot),
         'c.', label='the projection of class 2')

plt.plot(x_plot[:, 0], y_plot, 'k', label='projection line')
plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.show()
