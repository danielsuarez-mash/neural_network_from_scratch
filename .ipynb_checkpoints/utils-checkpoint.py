import numpy as np


def relu(Z):

    relu = np.maximum(0, Z)

    return relu


def sigmoid(Z):

    sigmoid = np.divide(1, 1 + np.exp(-Z))

    return sigmoid


def softmax(Z):

    expZ = np.exp(Z - np.max(Z))

    return expZ / expZ.sum(axis=0, keepdims=True)
