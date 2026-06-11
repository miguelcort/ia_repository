"""
Lección: 04-reconocimiento-de-habla-asr
Fase: 06
Prerrequisitos: 03-clasificacion-de-audio
"""
from __future__ import annotations
import sys
import numpy as np


def wer(reference, hypothesis):
    """Word Error Rate: edit distance / numero de palabras en reference.
    Edit distance = substitutions + insertions + deletions."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    # Levenshtein distance DP
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],     # delete
                    dp[i][j - 1],     # insert
                    dp[i - 1][j - 1]  # substitute
                )
    if m == 0:
        return 0.0 if n == 0 else 1.0
    return dp[m][n] / m


def cer(reference, hypothesis):
    """Character Error Rate."""
    ref_chars = list(reference.replace(" ", ""))
    hyp_chars = list(hypothesis.replace(" ", ""))
    m, n = len(ref_chars), len(hyp_chars)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    if m == 0:
        return 0.0
    return dp[m][n] / m


def tokenizar_caracteres(texto):
    return list(texto.replace(" ", "_"))


def mock_asr_decode(mel_spectrogram, n_steps=20):
    """Mock: ASR decoder (CTC o attention). Devuelve texto."""
    rng = np.random.default_rng(0)
    vocab = ["hola", "mundo", "el", "gato", "come"]
    n_frames = mel_spectrogram.shape[1] if mel_spectrogram.ndim > 1 else 100
    # Mock: emitir n_steps tokens
    tokens = [vocab[i % len(vocab)] for i in range(min(n_steps, n_frames // 5))]
    return " ".join(tokens)


def main() -> int:
    ref = "el gato come pescado"
    hyp = "el gato come"
    print(f"WER: {wer(ref, hyp):.3f}")
    print(f"CER: {cer(ref, hyp):.3f}")
    # Mock ASR
    mel = np.random.default_rng(0).normal(size=(80, 100))
    out = mock_asr_decode(mel)
    print(f"ASR output: {out[:50]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())