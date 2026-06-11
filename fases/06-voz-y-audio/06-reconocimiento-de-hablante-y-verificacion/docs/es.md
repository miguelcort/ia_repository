# Reconocimiento de hablante y verificación

> Biometria vocal. x-vector, ECAPA-TDNN, TitaNet. Verification (1:1) e identification (1:N). SOTA: EER < 1% en VoxCeleb. Diarization: pyannote.audio, NeMo Sortformer. Frameworks: SpeechBrain, NeMo, resemblyzer.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 04-reconocimiento-de-habla-asr
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar x-vector embedding mock.
- Calcular cosine similarity.
- Implementar verification y identification.
- Calcular EER.

## Constrúyelo

```python
def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0: return 0.0
    return float(a @ b / (na * nb))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-speaker-id
fase: 06
leccion: 06
---

1. Verification: ECAPA-TDNN + cosine + threshold.
2. Identification: ECAPA vs N embeddings.
3. Diarization: pyannote.audio, NeMo Sortformer.
4. Anti-spoofing: RawNet, AASIST.
5. Edge: ResNetSE, TitaNet.
6. EER, minDCF, VoxCeleb eval.
```

## Ejercicios

1. **resemblyzer**: usar resemblyzer para voice embedding.
2. **pyannote diarization**: aplicar pyannote a un audio
   con multiples hablantes.
3. **Desafio**: sistema de verification con ECAPA-TDNN,
   PLDA, anti-spoofing, eval EER en golden set.

## Lecturas recomendaciones

- "x-vectors" (Snyder et al., 2018)
- "ECAPA-TDNN" (Desplanques et al., 2020)
- SpeechBrain: <https://speechbrain.github.io/>
- NeMo: <https://github.com/NVIDIA/NeMo>

---

> 📚 **Adaptación al español** de la lección "[Speaker Recognition & Verification]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).