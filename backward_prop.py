import numpy as np
import utils


class BackwardProp:

    def __init__(self, AL, Y, caches):

        self.AL = AL
        self.Y = Y
        self.caches = caches

    def onehot_encoder(self, array):

        ohe = np.zeros((len(array), array.max() + 1))
        ohe[np.arange(len(array)), array] = 1
        ohe = ohe.transpose()

        return ohe

    def linear_backward(self, dZ, cache):

        A_prev, W, b = cache
        m = A_prev.shape[1]

        dA_prev = np.dot(W.T, dZ)
        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        return dA_prev, dW, db

    def linear_activation_backward(self, dA, cache, Y, activation_function):

        linear_cache, activation_cache = cache
        Z = activation_cache

        # calculation depends on activation function
        if activation_function == "sigmoid":

            dZ = dA * (utils.sigmoid(Z) * (1 - utils.sigmoid(Z)))
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation_function == "softmax":

            dZ = utils.softmax(Z) - Y
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        elif activation_function == "relu":

            dZ = np.array(dA, copy=True)
            dZ[Z <= 0] = 0
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)

        return dA_prev, dW, db

    def l_model_backward(self):
        """Going back through the layers, this function computes gradients for the loss function with respect to each neuron."""

        # number of layers
        L = len(self.caches)

        # gradient storage
        grads = dict()

        # one hot encode y
        Y_ohe = self.onehot_encoder(self.Y)

        # initiate backward propagation
        dAL = -np.divide(Y_ohe, self.AL) + np.divide(1 - Y_ohe, 1 - self.AL)

        # propagate backwards through last layer separately (as it has a different activation function)
        current_cache = self.caches[L - 1]
        grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = (
            self.linear_activation_backward(
                dAL, current_cache, Y_ohe, activation_function="softmax"
            )
        )

        # cycle through layers in reverse
        for l in reversed(range(1, L)):
            current_cache = self.caches[l - 1]
            grads["dA" + str(l - 1)], grads["dW" + str(l)], grads["db" + str(l)] = (
                self.linear_activation_backward(
                    grads["dA" + str(l)],
                    current_cache,
                    Y_ohe,
                    activation_function="relu",
                )
            )

        return grads
