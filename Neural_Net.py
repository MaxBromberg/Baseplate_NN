"""
Personal project recreating a basic neural net based on 3Blue1Brown's series as my keras install & C++ patience is down.
NN playlist: https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi,
minimal plagerism from Michael Nielsen's book, as seen in the video (linked below)
http://neuralnetworksanddeeplearning.com/chap1.html
"""

import numpy as np
import random
import activation_functions as af

random.seed(42)
# static functions


def cost(x, y):
    return (x-y) * (x-y)


def cost_prime(x, y):
    return 2*(x - y)


def compute_z(w, b, a):
    return np.einsum('ij, j', w, a) + b


class NeuralNet:
    def __init__(self, NN_dimensions, activation_fct: str, learning_rate=1e-4):
        self.NN_shape = NN_dimensions
        # Random initialization of (dense) weights and biases, hidden nodes.
        self.weights = [np.random.rand(self.NN_shape[layer+1], self.NN_shape[layer]) for layer in range(len(self.NN_shape[:-1]))]
        self.biases = [-np.random.rand(layer) for layer in self.NN_shape[1:]]
        self.nodes = [np.zeros(NN_dimensions[0]), *[np.random.rand(layer) for layer in self.NN_shape[1:-1]], np.empty(NN_dimensions[-1])]

        self.activation_fct = activation_fct
        self.AF = af.ActivationFunctions(a=0, b=1)
        self.sigma = self.AF.sigma_dict[activation_fct][0]
        self.sigma_prime = self.AF.sigma_dict[activation_fct][1]
        self.eta = learning_rate

    def set_activation_fct_parameters(self, a, b):
        self.AF = self.AF(a=a, b=b)
        self.sigma = self.AF.sigma_dict[self.activation_fct][0]
        self.sigma_prime = self.AF.sigma_dict[self.activation_fct][1]

    def feedforward(self, x):  # yields prediction from given input data based on weight state
        # NOT updates node values, i.e. [a^(i+1) = sigma( Wa^(i) + b )] for i < num_layers
        a = x
        for w, b in zip(self.weights, self.biases):
            a = self.sigma(compute_z(w, b, a))
        return a

    def evaluate(self, test_data):
        return np.sum([cost(self.feedforward(test_data[_][0]), test_data[_][1]) for _ in range(len(test_data[:]))]) / len(test_data[:])

    def SGD(self, training_data, epochs, mini_batch_size, test_data=None):
        """
        Stochastic Gradient Descent; trains the network based on a subset of the data
        :param training_data: full set of training data (dims: training samples X layer 1 dim)
        :param epochs: number of times the network will be trained
        :param mini_batch_size: size of the 'mini batch', e.g. (0.1) * len( training data)
        :param test_data: optional data to test against, giving r^2 error each batch if so (though slower)
        """
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, len(training_data), mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch)
            if test_data is not None:
                print(f"Epoch {epoch}: r^2 error: {np.round(self.evaluate(test_data), 5)}")
            else:
                print(f"Epoch {epoch} completed")

    def update_mini_batch(self, mini_batch):
        """Update Network weights and bias via gradient descent through back prop to a single mini-batch."""
        nabla_w = [np.zeros(self.weights[i].shape) for i in range(len(self.weights))]
        nabla_b = [np.zeros(self.biases[i].shape) for i in range(len(self.biases))]
        # Calculation of change to gradient for weights, biases
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backpropagation(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        # Actual update of weights, biases used by next mini batch update
        self.weights = [w - (self.eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (self.eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backpropagation(self, x, y):
        """x, y are training data with corresponding input and outputs
        returns delta nabla w & delta nabla b, gradient of weights and biases respectively """
        nabla_w = [np.zeros(self.weights[i].shape) for i in range(len(self.weights))]
        nabla_b = [np.zeros(self.biases[i].shape) for i in range(len(self.biases))]
        activations = [x]   # layer by layer activations
        zs = []   # layer by layer z (weighted sum wa+b) evaluations
        for w, b in zip(self.weights, self.biases):
            z = compute_z(w, b, activations[-1])
            zs.append(z)
            activations.append(self.sigma(z))
        # Backward Pass, now that all zs and node values (activations) are known
        delta = cost_prime(activations[-1], y) * self.sigma_prime(zs[-1])  # as delta is used in both nabla_w, nabla_b
        nabla_w[-1] = np.einsum('i..., j -> ij', delta, activations[-2])  # == vector mult. expansion to matrix, e.g. dims j, i -> j x i
        nabla_b[-1] = delta  # last layers directly calculated
        for layer in range(2, len(self.NN_shape)):
            delta = np.einsum('ji, j', self.weights[-layer+1], delta) * self.sigma_prime(zs[-layer])  # == np.dot(delta.T, activations[-2])
            nabla_w[-layer] = np.einsum('i..., j -> ij', delta, activations[-layer-1])
            nabla_b[-layer] = delta
        return nabla_w, nabla_b

    # Logs:
    def log_network_shapes(self):
        print(f"weights: {[self.weights[i].shape for i in range(len(self.weights))]} ")
        print(f"nodes: {[self.nodes[i].shape for i in range(len(self.nodes))]}")
        print(f"biases: {[self.biases[i].shape for i in range(len(self.biases))]}")



