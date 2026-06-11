"""
Lección: 05-analisis-de-sentimiento
Fase: 05
Prerrequisitos: 02-bolsa-de-palabras-y-tfidf
"""
from __future__ import annotations
import sys
import re
import numpy as np
from collections import defaultdict


def tokenizar(texto):
    """Tokenizacion basica."""
    return re.findall(r"\b\w+\b", texto.lower())


class LexiconSentiment:
    """Sentiment con lexicon: palabra -> score [-1, 1]."""
    def __init__(self, lexicon=None):
        if lexicon is None:
            lexicon = {
                "bueno": 0.8, "excelente": 1.0, "genial": 0.9,
                "malo": -0.8, "horrible": -1.0, "terrible": -0.9,
                "no": -0.5,  # negador
            }
        self.lexicon = lexicon

    def score_text(self, texto, manejo_negacion=True):
        """Suma scores de las palabras del lexicon. Opcionalmente maneja negacion."""
        tokens = tokenizar(texto)
        score = 0.0
        n = 0
        prev_neg = False
        for t in tokens:
            if t in self.lexicon:
                s = self.lexicon[t]
                if manejo_negacion and prev_neg:
                    s = -s
                score += s
                n += 1
            prev_neg = (t == "no" or t == "nunca")
        if n == 0:
            return 0.0
        return score / n

    def predecir(self, texto, umbral=0.0):
        s = self.score_text(texto)
        if s > umbral:
            return "positivo"
        elif s < -umbral:
            return "negativo"
        return "neutro"


class SentimentClasificador:
    """Mock: clasificador ML para sentiment."""
    def __init__(self):
        self.pesos = None
        self.vocab = None

    def fit(self, X, y, lr=0.1, epocas=50):
        """Entrena regresion logistica simple con BoW."""
        # Construir vocabulario
        from collections import Counter
        counter = Counter()
        for texto in X:
            counter.update(tokenizar(texto))
        self.vocab = {w: i for i, w in enumerate(counter.keys())}
        # BoW
        V = len(self.vocab)
        X_mat = np.zeros((len(X), V))
        for i, texto in enumerate(X):
            for t in tokenizar(texto):
                if t in self.vocab:
                    X_mat[i, self.vocab[t]] += 1
        # Targets
        y_bin = np.array([1 if label == "positivo" else 0 for label in y], dtype=float)
        # Pesos
        self.pesos = np.zeros(V)
        for _ in range(epocas):
            z = X_mat @ self.pesos
            pred = 1 / (1 + np.exp(-z))
            grad = X_mat.T @ (pred - y_bin) / len(y)
            self.pesos -= lr * grad
        return self

    def predecir(self, X, umbral=0.5):
        V = len(self.vocab)
        X_mat = np.zeros((len(X), V))
        for i, texto in enumerate(X):
            for t in tokenizar(texto):
                if t in self.vocab:
                    X_mat[i, self.vocab[t]] += 1
        z = X_mat @ self.pesos
        pred = 1 / (1 + np.exp(-z))
        return ["positivo" if p > umbral else "negativo" for p in pred]

    def score(self, X, y):
        y_pred = self.predecir(X)
        return float(np.mean([1 if a == b else 0 for a, b in zip(y_pred, y)]))


def main() -> int:
    lex = LexiconSentiment()
    print(f"'excelente producto': {lex.predecir('excelente producto')}")
    print(f"'no me gusta': {lex.predecir('no me gusta')}")
    # Clasificador
    X_train = ["excelente producto", "muy malo", "genial experiencia", "horrible servicio"]
    y_train = ["positivo", "negativo", "positivo", "negativo"]
    clf = SentimentClasificador()
    clf.fit(X_train, y_train, epocas=200)
    X_test = ["buen producto", "muy malo"]
    y_test = ["positivo", "negativo"]
    acc = clf.score(X_test, y_test)
    print(f"Accuracy: {acc:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())