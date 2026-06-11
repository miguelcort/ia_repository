# Audio transformers (Whisper)

> Whisper: encoder-decoder transformer para ASR multilingual multitask. Pipeline: audio 16kHz → log-mel (80 channels, 25ms window, 10ms hop) → conv embedding stride-2 → encoder → decoder autoregresivo con cross-attention. Multitask: transcribe, translate, language ID, timestamps. Sizes: tiny→large-v3. Variantes: wav2vec 2.0 (self-supervised), HuBERT, Conformer.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/05-transformer-completo, 06-voice-and-audio
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar log-mel spectrogram.
- Implementar conv embedding stride-2.
- Diagnosticar Whisper vs wav2vec 2.0.
- Aplicar Whisper para ASR.

## Constrúyelo

```python
def log_mel_spectrogram(audio, n_mels=80, hop=160):
    n = len(audio)
    pad = (hop - n % hop) % hop
    audio = np.concatenate([audio, np.zeros(pad)])
    n_frames = len(audio) // hop
    frames = audio[:n_frames * hop].reshape(n_frames, hop)
    energy = np.abs(frames).mean(axis=-1)
    mel = mel_basis * energy.reshape(1, -1)
    return np.log(mel + 1e-6).T
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-whisper
fase: 07
leccion: 10
---

1. Audio -> log-mel 80.
2. Conv stride-2 -> encoder -> decoder.
3. Multitask: transcribe, translate, lang ID.
4. Fine-tuning: LoRA, lr 1e-5.
5. Production: faster-whisper, CTranslate2.
```

## Ejercicios

1. **Conformer**: implementar Conformer (CNN
   + attention) para ASR.
2. **LoRA fine-tune**: usar PEFT para fine-tune
   Whisper-small en espanol.
3. **Desafio**: streaming ASR con sliding
   window y overlapping chunks.

## Lecturas recomendadas

- "Robust Speech Recognition via Large-Scale Weak Supervision" (Radford et al., 2022)
- "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations" (Baevski et al., 2020)
- "Conformer: Convolution-augmented Transformer for Speech Recognition" (Gulati et al., 2020)

---

> 📚 **Adaptación al español** de la lección "[Audio Transformers Whisper]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).