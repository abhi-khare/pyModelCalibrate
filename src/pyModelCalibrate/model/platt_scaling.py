import numpy as np
from sklearn.linear_model import LogisticRegression


class PlattCalibrator:
    def __init__(self, probs: list, labels: list):

        # check for errors
        self.check_errors(probs, labels)
        # platt scaling model uses logistic regression model
        base_model = LogisticRegression(C=1e10, solver='lbfgs')
        eps = 1e-12
        np_probs = np.array(probs).astype(dtype=np.float64)
        np_probs = np.expand_dims(np_probs, axis=-1)
        np_probs = np.clip(np_probs, eps, 1 - eps)
        np_probs = np.log(np_probs / (1 - np_probs))
        self.labels = labels

    @staticmethod
    def check_errors(probs: list, labels: list, partition_scheme: str, kwargs: dict):

        assert len(probs) == len(labels), \
            f'Size mismatch. probs array contains {len(probs)} elements \
                label array contains {len(labels)} element'

        assert len(probs) != 0, f"probs array must contain atleast 1 element"

        assert len(labels) != 0, f"label array must contain atleast 1 element"

        for prob in probs:
            if not isinstance(prob, (float,)):
                raise ValueError(f"variable prob contains value of incorrect datatype."
                                 f"Expected float found {type(prob)}")

            if not 0 <= prob <= 1:
                raise ValueError(f"Value out of Bound. Expected value between 0 and 1.")

        for label in labels:
            if not isinstance(label, (int,)):
                raise ValueError(f"variable label contains value of incorrect datatype."
                                 f"Expected float found {type(label)}")

    def fit(self):

        self.base_model.fit(self.np_probs, self.labels)

    def predict(self, probs):
        x = np.array(probs, dtype=np.float64)
        x = np.clip(x, self.eps, 1 - self.eps)
        x = np.log(x / (1 - x))
        x = x * self.base_model.coef_[0] + self.base_model.intercept_
        output = 1 / (1 + np.exp(-x))
        return output
