import Neural_Net as NN
import numpy as np
from matplotlib import pyplot as plt

virtual_samples = 100
layer_dims = [100, 20, 8, 2]
test_net = NN.NeuralNet(layer_dims, "logistic")
test_net.log_network_shapes()
# print(test_net.feedforward(x=np.random.rand(test_net.NN_shape[0])))
# test_net.backpropagation(x=np.random.rand(test_net.NN_shape[0]), y=np.random.rand(test_net.NN_shape[-1]))

fake_data = np.array([(np.random.rand(test_net.NN_shape[0]), np.random.rand(test_net.NN_shape[-1])) for _ in range(virtual_samples)], dtype=object)
fake_test_data = np.array([(np.random.rand(test_net.NN_shape[0]), np.random.rand(test_net.NN_shape[-1])) for _ in range(virtual_samples)], dtype=object)
test_net.SGD(fake_data, epochs=10, mini_batch_size=int(0.1*fake_data.shape[0]), test_data=fake_test_data)

