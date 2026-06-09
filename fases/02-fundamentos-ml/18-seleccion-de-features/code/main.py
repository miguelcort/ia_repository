"""
Lección: 18-seleccion-de-features
Fase: 02
Prerrequisitos:** 06-preparacion-de-datos-y-feature-engineering
"""
from __future__ import annotations
import sys
import numpy as np


def varianza_threshold(X, umbral=0.0):
    """Elimina features con varianza <= umbral (casi constantes)."""
    var = X.var(axis=0)
    mascara = var > umbral
    return mascara, var


def correlacion_pearson(X):
    """Matriz de correlacion de Pearson."""
    Xc = X - X.mean(axis=0)
    norma = np.sqrt((Xc ** 2).sum(axis=0))
    norma = np.where(norma == 0, 1.0, norma)
    Xn = Xc / norma
    return Xn.T @ Xn


def seleccionar_por_correlacion(X, umbral=0.95):
    """Elimina features altamente correlacionadas entre si.
    Conserva la primera; elimina las siguientes que superen umbral."""
    C = np.abs(correlacion_pearson(X))
    np.fill_diagonal(C, 0)
    n = X.shape[1]
    eliminadas = set()
    for i in range(n):
        if i in eliminadas:
            continue
        for j in range(i + 1, n):
            if j in eliminadas:
                continue
            if C[i, j] > umbral:
                eliminadas.add(j)
    mascara = np.array([k not in eliminadas for k in range(n)])
    return mascara, eliminadas


def mutual_info_simple(x_discrete, y_discrete):
    """Mutual information discreta: I(X;Y) = sum p(x,y) log p(x,y) / (p(x) p(y)).
    Mas alta = mas dependencia."""
    n = len(x_discrete)
    valores_x, counts_x = np.unique(x_discrete, return_counts=True)
    valores_y, counts_y = np.unique(y_discrete, return_counts=True)
    p_x = dict(zip(valores_x, counts_x / n))
    p_y = dict(zip(valores_y, counts_y / n))
    # Conjunta
    xy = list(zip(x_discrete, y_discrete))
    from collections import Counter
    conjunta = Counter(xy)
    mi = 0.0
    for (xi, yi), c in conjunta.items():
        p_xy = c / n
        if p_xy > 0 and p_x[xi] > 0 and p_y[yi] > 0:
            mi += p_xy * np.log(p_xy / (p_x[xi] * p_y[yi]))
    return float(mi)


def forward_selection(X, y, scoring_fn, max_features=None):
    """Forward selection: empieza vacio, anade la feature que mas mejora.
    scoring_fn(features_idx, X, y) -> float (mayor mejor)."""
    n_features = X.shape[1]
    if max_features is None:
        max_features = n_features
    seleccionadas = []
    restantes = list(range(n_features))
    mejor_score_global = -np.inf
    while restantes and len(seleccionadas) < max_features:
        mejor_feat = None
        mejor_score = -np.inf
        for f in restantes:
            candidatas = seleccionadas + [f]
            s = scoring_fn(candidatas, X, y)
            if s > mejor_score:
                mejor_score = s
                mejor_feat = f
        if mejor_feat is None:
            break
        seleccionadas.append(mejor_feat)
        restantes.remove(mejor_feat)
        mejor_score_global = mejor_score
    return seleccionadas, mejor_score_global


def main() -> int:
    rng = np.random.default_rng(0)
    # 5 features: 3 utiles, 2 ruido
    X = rng.normal(size=(100, 5))
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    # Varianza
    mascara, var = varianza_threshold(X, umbral=0.5)
    print(f"Varianza: {var}, mascara: {mascara}")
    # Correlacion: anado una feature identica a la 0
    X2 = np.hstack([X, X[:, [0]]])
    mascara2, elim = seleccionar_por_correlacion(X2, umbral=0.95)
    print(f"Correlacion: eliminadas={elim}, mascara={mascara2}")
    # MI de cada feature
    for j in range(5):
        # Discretizamos en 3 bins
        x_disc = np.digitize(X[:, j], np.percentile(X[:, j], [33, 66]))
        mi = mutual_info_simple(x_disc, y)
        print(f"  Feature {j}: MI = {mi:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())