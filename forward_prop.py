import numpy as np
import streamlit as st
import utils


class ForwardProp:

    def __init__(self, parameters):

        self.parameters = parameters

    def linear_forward(self, W, A, b):

        Z = np.dot(W, A) + b

        cache = (A, W, b)

        return Z, cache

    def linear_activation_forward(self, A_prev, W, b, activation):

        # compute Z
        Z, linear_cache = self.linear_forward(W, A_prev, b)

        # compute A
        if activation == "relu":
            A = utils.relu(Z)
        elif activation == "sigmoid":
            A = utils.sigmoid(Z)
        elif activation == "softmax":
            A = utils.softmax(Z)

        # caches
        activation_cache = Z
        cache = (linear_cache, activation_cache)

        return A, cache

    def l_model_forward(self, X):
        """
        Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

        Arguments:
        X -- data, numpy array of shape (input size, number of examples)
        parameters -- output of initialize_parameters_deep()

        Returns:
        AL -- activation value from the output (last) layer
        caches -- list of caches containing:
                    every cache of linear_activation_forward() (there are L of them, indexed from 0 to L-1)
        """
        # find number of layers
        L = len(self.parameters) // 2

        # get ready to capture all caches from each layer
        caches = []

        # initialise A_prev as X
        A_prev = X

        # forward propagation for layers (aside from last)
        for l in range(1, L):

            # compute forward linear activation
            A, cache = self.linear_activation_forward(
                A_prev,
                self.parameters["W" + str(l)],
                self.parameters["b" + str(l)],
                activation="relu",
            )

            # reset A_prev
            A_prev = A

            caches.append(cache)

        # forward propagation for last layer
        self.AL, cache = self.linear_activation_forward(
            A_prev,
            self.parameters["W" + str(L)],
            self.parameters["b" + str(L)],
            activation="softmax",
        )
        caches.append(cache)

        return caches
