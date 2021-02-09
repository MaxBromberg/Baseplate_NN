import numpy as np


class ActivationFunctions:
    def __init__(self, a=0, b=1):
        """
        Choice activation functions:
        "linear", "relu", "logistic"
        a = intercept, b = slope
        ...A bit of accidental class here
        """
        self.a = a
        self.b = b
        self.sigma_dict = {
            "linear": [self.linear, self.linear_prime],
            "relu": [self.relu, self.relu_prime],
            "logistic": [self.logistic, self.logistic_prime]
        }

    def linear(self, x):
        return self.a + (self.b * x)

    def linear_prime(self):
        return self.b

    def relu(self, x):
        return np.maximum(np.zeros(x.size), self.a + (self.b * x))

    def relu_prime(self, x):
        return np.where(x > 0, self.a, 0.0)

    def logistic(self, x):
        return 1 / (1 + np.exp(-self.a - x * self.b))

    def logistic_prime(self, x):
        return self.logistic(x) * (1 - self.logistic(x))


