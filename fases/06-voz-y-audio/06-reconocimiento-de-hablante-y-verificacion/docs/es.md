# 06 — Reconocimiento y verificación de hablante

> Reconocimiento de hablante (¿quién está hablando?) y verificación (¿es este audio de Juan?). Es la base de VoiceID, sistemas de seguridad, y segmentación de hablantes en reuniones.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-clasificacion-de-audio
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar verificación de hablante con embeddings
  (x-vectors, d-vectors).
- Aplicar ECAPA-TDNN y resemblyzer.
- Diagnosticar métricas: EER, minDCF.
- Conocer aplicaciones: VoiceID, diarización,
  autenticación.

## El problema

El reconocimiento de hablante identifica quién está
hablando de un conjunto conocido (close-set). La
verificación compara dos audios y decide si son del mismo
hablante (open-set, decisión binaria). Es la base de VoiceID
(autenticación por voz), sistemas de seguridad, y
diarización (segmentar reunión por hablante).

## El concepto

**Embeddings de hablante.** Un audio de voz se mapea a
un vector denso (embedding) que captura la identidad del
hablante. Variantes:

- **d-vector:** LSTM entrenado con clasificación de
  hablante. El embedding es la penúltima capa.
- **x-vector (Snyder et al., 2018):** TDNN con pooling
  estadístico (mean + std). Más robusto que d-vector.
- **ECAPA-TDNN (Desplanques et al., 2020):** Channel-
  Aware, attentive pooling. SOTA en VoxCeleb.
- **WavLM / Whisper embeddings:** pueden usarse para
  verificación, no específicamente entrenados.

**Verificación (open-set).**

1. Embed el audio de enrollment y el audio de test.
2. Calcular cosine similarity.
3. Threshold: si sim > threshold, es el mismo hablante.

**Métricas.**

- **EER (Equal Error Rate):** punto donde FAR = FRR.
  Menor mejor. SOTA en VoxCeleb: < 2%.
- **minDCF (minimum Detection Cost Function):** pondera
  FAR y FRR según costos de aplicación. Estándar en
  NIST SRE.

**Aplicaciones.**

- **VoiceID / autenticación:** comparar audio de
  enrollment con el de login. Más seguro que passwords
  en algunos aspectos (no se puede robar mirando).
- **Diarización:** separar "quién habla cuándo" en
  reuniones. pyannote.audio es SOTA.
- **Forensics:** identificar al hablante en una
  grabación legal.
- **Custom voice:** extraer embeddings para voice
  cloning.

**Trampas.**

- **Voz cambiada:** un hablante enfermo o en estado
  emocional diferente puede tener embedding distante.
- **Ruido de fondo:** degrada los embeddings. Usar
  VAD para limpiar.
- **Impostores con voz similar:** gemelos, doblajes.
  La biometría de voz no es 100% segura.

## Constrúyelo

```python
import numpy as np


def cosine_similarity(emb1, emb2):
    """Similitud coseno entre dos embeddings de hablante."""
    n1 = np.linalg.norm(emb1)
    n2 = np.linalg.norm(emb2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(emb1, emb2) / (n1 * n2))


def enroll_speaker(audio_embeddings):
    """Calcula el embedding promedio de enrollment."""
    return np.mean(audio_embeddings, axis=0)


def verify(emb_test, emb_enrollment, threshold=0.7):
    """Decide si el audio de test es del hablante enrolled."""
    sim = cosine_similarity(emb_test, emb_enrollment)
    return sim >= threshold, sim


def compute_eer(scores, labels):
    """Equal Error Rate: punto donde FAR = FRR."""
    # scores: similitud, labels: 1 = same, 0 = different
    thresholds = np.linspace(0, 1, 100)
    fars, frrs = [], []
    for t in thresholds:
        predictions = (scores >= t).astype(int)
        fp = np.sum((predictions == 1) & (labels == 0))
        fn = np.sum((predictions == 0) & (labels == 1))
        fars.append(fp / max(np.sum(labels == 0), 1))
        frrs.append(fn / max(np.sum(labels == 1), 1))
    eer = min(np.maximum(np.array(fars), np.array(frrs)))
    return eer, thresholds[np.argmin(np.abs(np.array(fars) - np.array(frrs)))]
```

## Úsalo

```bash
pip install resemblyzer
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-speaker-verification
fase: 06
leccion: 06
---

Eres un asistente que ayuda con reconocimiento/verifica-
ción de hablante. Recibirás el caso de uso. Tu trabajo:

1. Si quieres SOTA: ECAPA-TDNN o WavLM pre-entrenado
   en VoxCeleb.
2. Para producción simple: SpeechBrain (librería
   canónica).
3. Enrollment: 5-10 audios del hablante (al menos 30s
   en total).
4. Verificación: cosine similarity con threshold
   0.7-0.8.
5. Para diarización: pyannote.audio.
6. Evaluar con EER y minDCF sobre VoxCeleb test.
7. Advertir contra spoofing (audio grabado,
   síntesis de voz): añadir anti-spoofing.
```

## Ejercicios

1. **ECAPA-TDNN**: usa SpeechBrain para extraer
   embeddings.
2. **Diarización**: aplica pyannote.audio a una
   reunión grabada.
3. **Desafío**: implementa verificación con
   detección de spoofing.

## Lecturas recomendadas

- *x-vectors* — Snyder et al., 2018.
- *ECAPA-TDNN* — Desplanques et al., 2020.
- *VoxCeleb*: <http://www.robots.ox.ac.uk/~vgg/data/voxceleb>.
- SpeechBrain: <https://speechbrain.github.io>.
- pyannote-audio: <https://github.com/pyannote/pyannote-audio>.

---

> 📚 **Adaptación al español** de la lección "[Speaker Recognition and Verification]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
