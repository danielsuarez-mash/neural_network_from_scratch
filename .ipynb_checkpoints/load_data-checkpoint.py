import numpy as np
import pandas as pd

class DataLoader:

    def __init__(self, filepath):

        self.filepath = filepath

    def prepare_data(self):

        # load data
        train = pd.read_csv(self.filepath + "train.csv")
        test = pd.read_csv(self.filepath + "test.csv")

        # get feature data
        X_train = train.drop(columns=["label"]).transpose()

        # get target variable
        y_train = train["label"]

        # convert to numpy
        X_train = X_train.to_numpy()
        y_train = y_train.to_numpy()
        test = test.to_numpy()

        return X_train, y_train, test
