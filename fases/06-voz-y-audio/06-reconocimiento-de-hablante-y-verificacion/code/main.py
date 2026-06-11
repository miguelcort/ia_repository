"""
Lección: 06-reconocimiento-de-hablante-y-verificacion
Fase: 06
Prerrequisitos: 04-reconocimiento-de-habla-asr
"""
from __future__ import annotations
import sys
import numpy as np


def embedding_hablante_mock(senal, dim=192):
    """Mock: x-vector / d-vector / ECAPA embedding.
    En produccion: SpeechBrain, NeMo, ResNetSE."""
    rng = np.random.default_rng(hash(senal.tobytes()[:50]) % 2**32)
    return rng.normal(0, 1, size=(dim,))


def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def verificacion_hablante(emb_audio, emb_referencia, umbral=0.7):
    """Speaker verification: 1 si misma persona, 0 si no.
    Cosine similarity + threshold."""
    sim = cosine_similarity(emb_audio, emb_referencia)
    return 1 if sim >= umbral else 0, sim


def identificacion_hablante(emb_audio, embeddings_db, etiquetas):
    """Speaker identification: identifica quien habla.
    embeddings_db: lista de embeddings pre-registrados. etiquetas: nombres."""
    sims = [cosine_similarity(emb_audio, e) for e in embeddings_db]
    idx = int(np.argmax(sims))
    return etiquetas[idx], float(sims[idx])


def equal_error_rate(scores, etiquetas, positivos):
    """EER: punto donde FAR = FRR.
    Mock: computa sobre scores ordenados."""
    # Simplificado: solo devolvemos el umbral optimo
    far_frr = []
    for thr in np.linspace(min(scores), max(scores), 100):
        tp = sum(1 for s, p in zip(scores, positivos) if p and s >= thr)
        fn = sum(1 for s, p in zip(scores, positivos) if p and s < thr)
        fp = sum(1 for s, p in zip(scores, positivos) if not p and s >= thr)
        tn = sum(1 for s, p in zip(scores, positivos) if not p and s < thr)
        far = fp / max(1, fp + tn)
        frr = fn / max(1, fn + tp)
        far_frr.append((thr, abs(far - frr)))
    if not far_frr:
        return 0.0, 0.0
    best = min(far_frr, key=lambda x: x[1])
    return float(best[0]), float(best[1])  # threshold, eer


def main() -> int:
    rng = np.random.default_rng(0)
    senal_a = rng.normal(size=16000)
    senal_b = rng.normal(size=16000)
    emb_a = embedding_hablante_mock(senal_a)
    emb_b = embedding_hablante_mock(senal_b)
    sim = cosine_similarity(emb_a, emb_b)
    print(f"Similitud: {sim:.3f}")
    res, score = verificacion_hablante(emb_a, emb_b, umbral=0.5)
    print(f"Verificacion (umbral 0.5): {res} (sim={score:.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())