"""
Lección: 08-clonacion-de-voz-y-conversion
Fase: 06
Prerrequisitos: 07-texto-a-habla-tts
"""
from __future__ import annotations
import sys
import numpy as np


def speaker_encoder_mock(audio, dim=192):
    """Mock: encoder de speaker (x-vector style)."""
    rng = np.random.default_rng(hash(audio.tobytes()[:50]) % 2**32)
    return rng.normal(0, 1, size=(dim,))


def voice_conversion_mock(source_audio, target_speaker_emb, output_dim=None):
    """Mock: convierte el audio source al speaker target.
    Conserva contenido linguistico, cambia timbre."""
    n_samples = len(source_audio) if output_dim is None else output_dim
    # Mock: senal source + ruido modulado por speaker
    rng = np.random.default_rng(0)
    factor = float(target_speaker_emb.mean())
    audio = source_audio * (1 + 0.1 * factor)
    audio = audio + 0.01 * rng.normal(size=len(audio))
    return audio[:n_samples]


def speaker_similarity(emb1, emb2):
    """Cosine similarity entre embeddings."""
    na, nb = np.linalg.norm(emb1), np.linalg.norm(emb2)
    if na == 0 or nb == 0:
        return 0.0
    return float(emb1 @ emb2 / (na * nb))


def speaker_consistency(generated_audios, ref_embedding):
    """Evalua consistencia del speaker en varios audios generados.
    Similitud promedio entre emb del audio generado y reference."""
    sims = []
    for audio in generated_audios:
        emb = speaker_encoder_mock(audio)
        sims.append(speaker_similarity(emb, ref_embedding))
    return float(np.mean(sims))


def main() -> int:
    sr = 16000
    source = np.random.default_rng(0).normal(size=sr)
    target_emb = speaker_encoder_mock(np.random.default_rng(1).normal(size=sr))
    converted = voice_conversion_mock(source, target_emb)
    print(f"Source: {source.shape}, Converted: {converted.shape}")
    sim_original = speaker_similarity(
        speaker_encoder_mock(source), target_emb
    )
    sim_converted = speaker_similarity(
        speaker_encoder_mock(converted), target_emb
    )
    print(f"Sim original: {sim_original:.3f}, sim converted: {sim_converted:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())