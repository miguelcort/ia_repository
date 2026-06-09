"""
Lección: 13-pipelines-de-ml-y-tracking
Fase: 02
Prerrequisitos:** 06-preparacion-de-datos-y-feature-engineering
"""
from __future__ import annotations
import sys
import json
import time
import hashlib
import numpy as np
from pathlib import Path


def estandarizar(X):
    """Centra y escala: (X - media) / std."""
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma, {"mu": mu.tolist(), "sigma": sigma.tolist()}


def imputar_media(X):
    """Sustituye NaN por la media de la columna."""
    mu = np.nanmean(X, axis=0)
    X_filled = X.copy()
    for j in range(X.shape[1]):
        X_filled[np.isnan(X[:, j]), j] = mu[j]
    return X_filled, {"mu": mu.tolist()}


class PipelineStep:
    """Paso individual: fit + transform."""
    def __init__(self, nombre, fn_fit_transform):
        self.nombre = nombre
        self.fn_fit_transform = fn_fit_transform
        self.estado = None

    def fit_transform(self, X):
        Xt, estado = self.fn_fit_transform(X)
        self.estado = estado
        return Xt

    def transform(self, X):
        # Re-uso: aplica la transformacion aprendida en fit
        if self.nombre == "estandarizar":
            mu = np.array(self.estado["mu"])
            sigma = np.array(self.estado["sigma"])
            return (X - mu) / sigma
        elif self.nombre == "imputar":
            mu = np.array(self.estado["mu"])
            X_filled = X.copy()
            for j in range(X.shape[1]):
                X_filled[np.isnan(X[:, j]), j] = mu[j]
            return X_filled
        return X


class Pipeline:
    """Encadena pasos secuenciales."""
    def __init__(self, pasos):
        self.pasos = pasos

    def fit_transform(self, X):
        for p in self.pasos:
            X = p.fit_transform(X)
        return X

    def transform(self, X):
        for p in self.pasos:
            X = p.transform(X)
        return X


def run_id(params, X_hash):
    """Genera un ID de corrida a partir de params + datos."""
    payload = json.dumps(params, sort_keys=True) + X_hash
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def log_run(log_path, run_id, params, score, duracion):
    """Anota una corrida a un JSONL."""
    entrada = {
        "run_id": run_id,
        "params": params,
        "score": score,
        "duracion_s": duracion,
        "ts": time.time(),
    }
    with open(log_path, "a") as f:
        f.write(json.dumps(entrada) + "\n")


def main() -> int:
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3))
    # Anade NaN
    X[0, 0] = np.nan
    X[5, 1] = np.nan
    p = Pipeline([
        PipelineStep("imputar", lambda X: imputar_media(X)),
        PipelineStep("estandarizar", lambda X: estandarizar(X)),
    ])
    Xt = p.fit_transform(X)
    print(f"Shape original: {X.shape}, transformado: {Xt.shape}")
    print(f"Media Xt: {Xt.mean(axis=0)}")
    print(f"Std Xt: {Xt.std(axis=0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())