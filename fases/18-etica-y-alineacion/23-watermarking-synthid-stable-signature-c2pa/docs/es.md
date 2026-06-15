# 23 — Watermarking: SynthID, Stable Signature, C2PA

> Watermarking de LLM/text: SynthID (Google DeepMind), Stable Signature (Meta), C2PA para media. Permite detectar AI-generated content. Trade-off: detectability vs robustness.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18/26
**Tiempo estimado:** ~25 minutos

## Objetivos

- Definir watermarking vs provenance.
- Implementar token-level watermark.
- Evaluar robustness a paraphrase.
- Diagnosticar detectability.

## Constrúyelo

```python
import hashlib
import numpy as np


def token_watermark(token_logits, key, vocab_size):
    """SynthID-style: sesgar logits según key.
    Detección: statistical test sobre bias."""
    rng = np.random.RandomState(key)
    bias = rng.randn(vocab_size)
    return token_logits + bias


def detect_watermark(tokens, key, vocab_size, threshold=0.5):
    """Statistical test: si los tokens son biased por key."""
    rng = np.random.RandomState(key)
    bias = rng.randn(vocab_size)
    scores = [bias[t] for t in tokens]
    avg = np.mean(scores)
    return avg > threshold


def c2pa_provenance(metadata, signature_key):
    """C2PA: criptographic signature en metadata."""
    payload = str(metadata).encode()
    return hashlib.sha256(signature_key + payload).hexdigest()
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-watermark
fase: 18
leccion: 23
---

1. SynthID-style token watermarking.
2. Statistical detection.
3. C2PA provenance para media.
4. Robustness a paraphrase.
```

## Ejercicios

1. **Watermark**: implementar en toy LM.
2. **Detect**: medir false positive.
3. **Desafío**: diseño robusto a
   paraphrase attack.

## Lecturas recommandée

- "SynthID Text" (Google DeepMind 2024)
- "Stable Signature" (Meta 2024)
- "C2PA Standard" (C2PA 2023)

---

> 📚 **Adaptación al español** de la lección
> "[23-watermarking-synthid-stable-signature-c2pa]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
