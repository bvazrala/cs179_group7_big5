"""Gaussian factor-analysis baseline.

Inputs are standardized; test fit is the mean Gaussian log-likelihood per
respondent (FactorAnalysis.score). conditional_mean predicts each item from
the remaining items under the fitted covariance; thresholding it on binary
data gives a masked-item accuracy comparable to the Ising model.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler


class FactorModel:
    def __init__(self, n_factors=5, seed=0, rotation=None):
        self.n_factors = n_factors
        self.seed = seed
        self.rotation = rotation  # None or "varimax"; affects loadings only,
        # not the fitted covariance or any likelihood/accuracy metric
        self.scaler = self.fa = self.cov = None

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.scaler = StandardScaler().fit(X)
        Z = self.scaler.transform(X)
        self.fa = FactorAnalysis(n_components=self.n_factors,
                                 rotation=self.rotation,
                                 random_state=self.seed).fit(Z)
        self.cov = self.fa.get_covariance()
        return self

    def test_log_likelihood(self, X, ridge=1e-6):
        """Mean per-respondent Gaussian log-likelihood on held-out data,
        computed from the fitted covariance via a Cholesky factorization with
        a small ridge. Equivalent to FactorAnalysis.score on well-conditioned
        data but avoids the precision-matrix overflow that score can hit when
        the covariance is near-singular."""
        Z = self.scaler.transform(np.asarray(X, dtype=float))
        d = self.cov.shape[0]
        S = self.cov + ridge * np.eye(d)
        L = np.linalg.cholesky(S)
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        Zc = Z - self.fa.mean_
        y = np.linalg.solve(L, Zc.T)
        quad = np.einsum("ij,ij->j", y, y)
        ll = -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)
        return float(np.mean(ll))

    def loadings(self, items=None):
        L = self.fa.components_.T  # (n_items, n_factors)
        if items is None:
            return L
        return pd.DataFrame(L, index=items,
                            columns=[f"F{k + 1}" for k in range(L.shape[1])])

    def conditional_mean(self, X):
        """E[x_i | x_{-i}] for every item under the fitted covariance,
        returned in original units."""
        Z = self.scaler.transform(np.asarray(X, dtype=float))
        C, mu = self.cov, self.fa.mean_
        d = C.shape[0]
        idx = np.arange(d)
        out = np.empty_like(Z)
        for i in range(d):
            r = idx != i
            beta = np.linalg.solve(C[np.ix_(r, r)], C[r, i])
            out[:, i] = mu[i] + (Z[:, r] - mu[r]) @ beta
        return self.scaler.inverse_transform(out)

    def masked_accuracy(self, X, threshold=0.5):
        """Accuracy of thresholded conditional means; meaningful when the
        model is fit on the binary matrix."""
        X = np.asarray(X, dtype=float)
        pred = self.conditional_mean(X) >= threshold
        return float(np.mean(pred == (X > 0.5)))
    