import numpy as np


class InitialiseParameters:

    def __init__(self, layer_dims):

        # network dimensions
        self.layer_dims = layer_dims

        # number of layers
        self.L = len(self.layer_dims)

    def initialise_parameters(self):

        # initialise params dictionary
        self.parameters = dict()

        # cycle through layers and initialise parameters
        for l in range(1, self.L):
            self.parameters["W" + str(l)] = (
                np.random.randn(self.layer_dims[l], self.layer_dims[l - 1]) * 0.01
            )
            self.parameters["b" + str(l)] = np.zeros((self.layer_dims[l], 1))

        return self.parameters
