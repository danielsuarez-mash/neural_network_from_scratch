import numpy as np


class ParameterUpdater:
    def __init__(self, parameters, grads, learning_rate):

        self.parameters = parameters
        self.grads = grads
        self.learning_rate = learning_rate

    def update_parameters(self):

        L = len(self.parameters) // 2

        # cycle through each layer and change parameters
        for l in range(1, L + 1):
            self.parameters["W" + str(l)] = (
                self.parameters["W" + str(l)]
                - self.learning_rate * self.grads["dW" + str(l)]
            )
            self.parameters["b" + str(l)] = (
                self.parameters["b" + str(l)]
                - self.learning_rate * self.grads["db" + str(l)]
            )

        return self.parameters
