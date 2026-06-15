"""
Lección: 23-watermarking-synthid-stable-signature-c2pa
Fase: 18
Ética y alineación: 23 Watermarking Synthid Stable Signature C2Pa.
"""
from __future__ import annotations
import sys
import numpy as np

import hashlib
import numpy as np


def token_watermark(token_logits, key, vocab_size):
    rng = np.random.RandomState(key)
    bias = rng.randn(vocab_size)
    return token_logits + bias


def detect_watermark(tokens, key, vocab_size, threshold=0.0):
    rng = np.random.RandomState(key)
    bias = rng.randn(vocab_size)
    scores = [bias[t] for t in tokens]
    return np.mean(scores) > threshold


def c2pa_provenance(metadata, signature_key):
    payload = str(metadata).encode()
    return hashlib.sha256(signature_key + payload).hexdigest()



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== 23-watermarking-synthid-stable-signature-c2pa ===")
    print(f"Python {sys.version.split()[0]}")
    funcs = ['token_watermark', 'detect_watermark', 'c2pa_provenance']
    print(f"Funciones disponibles: {len(funcs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
