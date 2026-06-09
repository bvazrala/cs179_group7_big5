"""Pairwise binary (Ising) model learned by node-wise L1-regularized
logistic regression (pseudo-likelihood / neighborhood selection).

Prediction uses the per-node conditional models; the symmetrized weight
matrix J defines the learned graph structure.
"""

import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

_EPS = 1e-12
_SK = tuple(int(x) for x in sklearn.__version__.split(".")[:2])


def _l1_logreg(C):
    """L1-penalized logistic regression compatible with old and new
    scikit-learn (the `penalty` argument is deprecated in favor of
    `l1_ratio` in 1.8+)."""
    if _SK >= (1, 8):
        return LogisticRegression(l1_ratio=1.0, C=C, solver="liblinear",
                                  max_iter=2000)
    return LogisticRegression(penalty="l1", C=C, solver="liblinear",
                              max_iter=2000)


def _sigmoid(z):
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def bernoulli_log_likelihood(X, P):
    """Mean log P(x) for binary X under probabilities P."""
    P = np.clip(P, _EPS, 1 - _EPS)
    return float(np.mean(X * np.log(P) + (1 - X) * np.log(1 - P)))


def masked_accuracy_from_proba(X, P):
    """Accuracy of predicting each entry from its conditional probability."""
    return float(np.mean((P > 0.5) == (X > 0.5)))


class IsingModel:
    """Per-node L1 logistic regressions give W (node-wise weights) and b
    (biases); J = symmetrized W masked by the `symmetrize` rule."""

    def __init__(self, C=0.1, symmetrize="and"):
        self.C = C
        self.symmetrize = symmetrize  # "and": both directions nonzero; "or": either
        self.W = self.b = self.J = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        W = np.zeros((d, d))
        b = np.zeros(d)
        cols = np.arange(d)
        for i in range(d):
            y = X[:, i]
            if y.min() == y.max():  # constant column: bias-only node
                p = np.clip(y.mean(), 1e-6, 1 - 1e-6)
                b[i] = np.log(p / (1 - p))
                continue
            clf = _l1_logreg(self.C)
            clf.fit(X[:, cols != i], y)
            W[i, cols != i] = clf.coef_[0]
            b[i] = clf.intercept_[0]
        self.W, self.b = W, b
        nz = W != 0
        mask = (nz & nz.T) if self.symmetrize == "and" else (nz | nz.T)
        self.J = 0.5 * (W + W.T) * mask
        return self

    def conditional_proba(self, X):
        """P(x_i = 1 | x_{-i}) for every entry; diag(W) = 0 so a single
        matrix product covers all nodes."""
        X = np.asarray(X, dtype=float)
        return _sigmoid(X @ self.W.T + self.b)

    def pseudo_log_likelihood(self, X):
        return bernoulli_log_likelihood(np.asarray(X, dtype=float),
                                        self.conditional_proba(X))

    def masked_accuracy(self, X):
        return masked_accuracy_from_proba(np.asarray(X, dtype=float),
                                          self.conditional_proba(X))

    @property
    def n_edges(self):
        return int((np.triu(self.J, 1) != 0).sum())


class IndependentBaseline:
    """Bias-only model: P(x_i = 1) = training mean of item i."""

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.p = np.clip(X.mean(axis=0), 1e-6, 1 - 1e-6)
        return self

    def conditional_proba(self, X):
        return np.broadcast_to(self.p, np.asarray(X).shape)

    def pseudo_log_likelihood(self, X):
        return bernoulli_log_likelihood(np.asarray(X, dtype=float),
                                        self.conditional_proba(X))

    def masked_accuracy(self, X):
        return masked_accuracy_from_proba(np.asarray(X, dtype=float),
                                          self.conditional_proba(X))
    