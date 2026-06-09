"""
Lección: 14-naive-bayes
Fase: 02
Prerrequisitos:** 02-modelos-lineales-y-regresion-logistica
"""
from __future__ import annotations
import sys
import numpy as np


class GaussianNB:
    """Naive Bayes Gaussiano. Asume P(x_i | y) ~ N(mu_y_i, sigma_y_i)."""
    def __init__(self):
        self.clases = None
        self.mu = None  # shape: (n_clases, n_features)
        self.sigma = None

    def fit(self, X, y):
        self.clases = np.unique(y)
        n_clases = len(self.clases)
        n_features = X.shape[1]
        self.mu = np.zeros((n_clases, n_features))
        self.sigma = np.zeros((n_clases, n_features))
        for i, c in enumerate(self.clases):
            Xc = X[y == c]
            self.mu[i] = Xc.mean(axis=0)
            self.sigma[i] = Xc.std(axis=0) + 1e-9  # suavizado
        return self

    def _log_p_clase(self, X):
        """Log P(y=c | x) proporcional a log P(x | y=c) + log P(y=c)."""
        n = len(X)
        log_prior = np.log(1.0 / len(self.clases))
        log_p = np.zeros((n, len(self.clases)))
        for i, c in enumerate(self.clases):
            # log P(x | c) = sum log N(x_j; mu_c_j, sigma_c_j)
            mu = self.mu[i]
            sigma = self.sigma[i]
            log_lik = -0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * ((X - mu) ** 2) / (sigma ** 2)
            log_p[:, i] = log_lik.sum(axis=1) + log_prior
        return log_p

    def predict(self, X):
        log_p = self._log_p_clase(X)
        idx = np.argmax(log_p, axis=1)
        return self.clases[idx]


class MultinomialNB:
    """Naive Bayes Multinomial. Para conteos (e.g. bag of words)."""
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.clases = None
        self.log_prior = None
        self.log_p_feature = None  # log P(x_j | y=c)

    def fit(self, X, y):
        self.clases = np.unique(y)
        n_clases = len(self.clases)
        n_features = X.shape[1]
        self.log_prior = np.zeros(n_clases)
        self.log_p_feature = np.zeros((n_clases, n_features))
        for i, c in enumerate(self.clases):
            Xc = X[y == c]
            self.log_prior[i] = np.log(len(Xc) / len(X))
            # Laplace smoothing
            conteo = Xc.sum(axis=0) + self.alpha
            self.log_p_feature[i] = np.log(conteo / conteo.sum())
        return self

    def predict(self, X):
        log_p = X @ self.log_p_feature.T + self.log_prior
        idx = np.argmax(log_p, axis=1)
        return self.clases[idx]


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def main() -> int:
    rng = np.random.default_rng(0)
    # Datos gaussianos separables
    X = np.vstack([
        rng.normal(loc=[-1, -1], scale=0.5, size=(30, 2)),
        rng.normal(loc=[1, 1], scale=0.5, size=(30, 2)),
    ])
    y = np.array([0] * 30 + [1] * 30)
    modelo = GaussianNB()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)
    print(f"GaussianNB accuracy: {accuracy(y, y_pred):.3f}")
    # MultinomialNB
    Xc = rng.integers(0, 5, size=(60, 4))
    yc = (Xc.sum(axis=1) > 8).astype(int)
    m = MultinomialNB()
    m.fit(Xc, yc)
    print(f"MultinomialNB accuracy: {accuracy(yc, m.predict(Xc)):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())