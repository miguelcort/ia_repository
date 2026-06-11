"""
Lección: 16-pipeline-de-vision-capstone
Fase: 04
Prerrequisitos: 15-vision-en-tiempo-real-en-borde
"""
from __future__ import annotations
import sys
import numpy as np


def stage_inference(imgs, batch_size=4):
    """Agrupa imagenes en batches y simula inference."""
    n = len(imgs)
    batches = [imgs[i:i+batch_size] for i in range(0, n, batch_size)]
    outputs = []
    for batch in batches:
        # Mock: cada imagen produce una salida (1, 1000) de logits
        out = np.random.default_rng(len(batch)).normal(0, 1, size=(len(batch), 1000))
        outputs.append(out)
    return np.vstack(outputs)


def stage_postprocess(logits, top_k=5):
    """Softmax + top-k. Devuelve [(clase, score), ...] por muestra."""
    exp = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    resultados = []
    for prob in probs:
        idx = np.argsort(prob)[::-1][:top_k]
        resultados.append([(int(i), float(prob[i])) for i in idx])
    return resultados


def latency_p50_p99(latencias):
    """Calcula p50 y p99 de latencias en ms."""
    arr = np.array(latencias)
    return {
        "p50_ms": float(np.percentile(arr, 50)),
        "p99_ms": float(np.percentile(arr, 99)),
        "mean_ms": float(arr.mean()),
    }


def throughput(lotes_por_segundo, batch_size):
    """Imagenes por segundo."""
    return lotes_por_segundo * batch_size


def main() -> int:
    imgs = [np.random.default_rng(i).normal(size=(224, 224, 3)) for i in range(8)]
    logits = stage_inference(imgs, batch_size=4)
    print(f"Logits: {logits.shape}")
    top5 = stage_postprocess(logits, top_k=3)
    print(f"Top-3 muestra 0: {top5[0]}")
    latencias = [10.5, 12.3, 11.0, 15.2, 9.8, 14.1, 13.5, 50.0, 11.2, 10.9]
    stats = latency_p50_p99(latencias)
    print(f"Latencia: {stats}")
    print(f"Throughput: {throughput(50, batch_size=4)} img/s")
    return 0


if __name__ == "__main__":
    sys.exit(main())