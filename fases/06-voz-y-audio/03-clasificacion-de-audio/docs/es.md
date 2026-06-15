# 03 — Clasificación de audio

> Clasificar sonidos: speech vs music vs noise, environmental sounds (ESC-50), music genre, instrument recognition. Es la base de muchos productos (Shazam, Alexa).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-fundamentos-de-audio
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar un clasificador de audio con features
 手工 (MFCC + clasificador).
- Aplicar AST (Audio Spectrogram Transformer) y PANN.
- Diagnosticar métricas: accuracy, F1 macro.
- Conocer datasets canónicos: ESC-50, AudioSet, UrbanSound8K.

## El problema

Clasificar un audio en categorías: speech vs music vs
silence (Alexa), environmental sound (alarm, dog bark,
glass breaking), music genre, instrument, etc. Es la base
de muchos productos: Shazam clasifica clips, Alexa detecta
palabras de activación, aplicaciones médicas clasifican
sonidos cardiopulmonares.

## El concepto

**Features tradicionales.** MFCC, mel-spectrograma,
spectral centroid, zero crossing rate, RMS energy. Se
extraen con librosa o torchaudio. Se alimentan a un
clasificador clásico (SVM, Random Forest, XGBoost).

**Datasets canónicos.**

- **ESC-50:** 50 clases de environmental sounds (2k
  clips, 5s cada uno).
- **AudioSet:** 527 clases de sonidos (Google,
  2M clips etiquetados).
- **UrbanSound8K:** 10 clases de sonidos urbanos
  (sirenas, tráfico, etc.).
- **GTZAN:** 10 géneros musicales (1k clips de 30s).
- **ESC-10:** subset pequeño de ESC-50 para prototipado.

**Modelos neuronales para audio.**

- **1D CNN:** convolución 1D sobre waveform crudo. Sorprendentemente efectivo.
- **2D CNN:** convolución 2D sobre mel-spectrograma. Base
  de muchos modelos.
- **PANN (Pre-trained Audio Neural Networks):** CNN
  pre-entrenada en AudioSet. CNN14, ResNet38, etc.
- **AST (Audio Spectrogram Transformer):** ViT
  aplicado a mel-spectrograma. SOTA en clasificación.
- **Whisper encoder:** puede usarse para audio
  classification (sin la cabeza de generación).
- **BEATs (Audio Pre-Training with Acoustic Tokenizers,
  Microsoft, 2023):** SOTA en audio classification.

**Pipeline típico.**

1. Cargar audio, resamplear a 16kHz mono.
2. Calcular log-mel spectrograma (64 o 128 bins).
3. Pasar al modelo pre-entrenado (AST, PANN, BEATs).
4. Aplicar cabeza de clasificación (linear).
5. Fine-tune con learning rate bajo (1e-4) en tu
   dataset.

**Trampas.**

- **Audio imbalance:** la mayoría de datasets tienen
  clases desbalanceadas. Usar class weights o
  oversampling.
- **Background noise:** los modelos pueden confundir
  "ruido" con "no clasificado". Hacer data augmentation
  con ruido (mixup, time stretch).
- **Diferentes sample rates:** los modelos esperan
  16kHz o 32kHz. Resamplear siempre.

## Constrúyelo

```python
import numpy as np


def extract_features(waveform, sr=16000, n_mfcc=40):
    """Extrae MFCC + delta + delta-delta como features."""
    # STFT simplificada
    n_fft, hop = 400, 160
    spec = np.abs(np.fft.rfft(waveform, n=n_fft))
    # Banco de filtros Mel simplificado
    mel_spec = spec[:n_mfcc]  # placeholder
    log_mel = np.log(np.maximum(mel_spec, 1e-10))
    # Delta y delta-delta (diferencias temporales)
    delta = np.diff(log_mel, axis=-1, prepend=log_mel[..., :1])
    delta2 = np.diff(delta, axis=-1, prepend=delta[..., :1])
    return np.concatenate([log_mel, delta, delta2], axis=0)


def audio_classifier_batch(features_list, labels, n_classes):
    """Clasificador simple: SVM-like con cosine similarity
    entre centroides."""
    # Calcular centroide por clase
    centroids = np.zeros((n_classes, features_list[0].shape[0]))
    for c in range(n_classes):
        class_feats = [f for f, l in zip(features_list, labels) if l == c]
        if class_feats:
            centroids[c] = np.mean(class_feats, axis=0)
    # Normalizar
    centroids = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-10)
    # Predecir
    predictions = []
    for features in features_list:
        norm = features / (np.linalg.norm(features) + 1e-10)
        sims = centroids @ norm
        predictions.append(np.argmax(sims))
    return np.array(predictions)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-audio-classification
fase: 06
leccion: 03
---

Eres un asistente que ayuda con clasificación de audio.
Recibirás el dataset y la tarea. Tu trabajo:

1. Si dataset pequeño: MFCC + SVM o XGBoost.
2. Si quieres SOTA: AST o BEATs pre-entrenados.
3. AudioSet pre-entrenado, fine-tune en tu dataset.
4. Log-mel spectrograma como features.
5. Augmentation: time stretch, pitch shift, ruido.
6. AdamW con lr=1e-4 para fine-tuning.
7. Métrica: accuracy + F1 macro.
8. Advertir contra clases desbalanceadas.
```

## Ejercicios

1. **MFCC**: implementa y clasifica ESC-10.
2. **AST**: aplica AST pre-entrenado a tu dataset.
3. **Desafío**: fine-tunea BEATs en un dataset pequeño
   y compara con SVM+MFCC.

## Lecturas recomendadas

- *PANNs: Pre-trained Audio Neural Networks* — Kong et al.,
  2020.
- *AST: Audio Spectrogram Transformer* — Gong et al.,
  2021.
- *BEATs* — Chen et al., 2023.
- HuggingFace Audio: <https://huggingface.co/docs/transformers/tasks/audio_classification>.

---

> 📚 **Adaptación al español** de la lección "[Audio Classification]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
