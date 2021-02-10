import Neural_Net as NN
import utils
import numpy as np
from tensorflow.keras.datasets import mnist  # yes, a tf install, but just for the easy mnist import.
from matplotlib import pyplot as plt


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# Flattening 28x28 pixel digits
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
# One hot encoding Y data
Y_train = utils.one_hot_encode(Y_train)
Y_test = utils.one_hot_encode(Y_test)
training_data = utils.np_to_list_tuples(X_train, Y_train)
test_data = utils.np_to_list_tuples(X_test, Y_test)

mnist_layers = [X_train.shape[1], 64, 16, 10]
mnist_net = NN.NeuralNet(mnist_layers, "logistic")
mnist_net.SGD(training_data, epochs=50, mini_batch_size=50, test_data=test_data[9500:])
print(mnist_net.one_hot_evaluate(X_test, Y_test))

