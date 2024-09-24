import numpy as np


class Cost:

    def __init__(self, AL, y):

        self.AL = AL
        self.y = y

    def ohe(self, array):

        ohe = np.zeros((len(array), array.max() + 1))
        ohe[np.arange(len(array)), array] = 1
        ohe = ohe.transpose()

        return ohe

    def compute_cost(self):

        # one hot encode y
        y_ohe = self.ohe(self.y)

        # number of examples
        m = len(self.y)

        # categorical cross-entropy loss function
        self.cost = (-1 / m) * np.sum(y_ohe * np.log(self.AL))

        return self.cost
