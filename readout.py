import numpy as np

class RidgeReadout:
    def __init__(self, alpha=1e-3):
        self.alpha, self.W = alpha, None

    def fit(self, feats, targets):
        X = self._bias(feats)
        self.W = np.linalg.solve(X.T @ X + self.alpha * np.eye(X.shape[1]), X.T @ targets)
        return self

    def predict(self, feats):
        return self._bias(feats) @ self.W

    @staticmethod
    def _bias(feats):
        return np.concatenate([feats, np.ones((feats.shape[0], 1))], axis=1)
