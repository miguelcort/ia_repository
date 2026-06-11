# Arquitectura Whisper y fine-tuning

> Encoder-decoder transformer: log-mel -> encoder -> decoder -> texto. 99 idiomas, 680K h. Whisper-large-v3 (1.5B), Distil-Whisper (39M, edge), faster-whisper (4x speedup). Fine-tune con Common Voice o tu data, push to Hub.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 04-reconocimiento-de-habla-asr
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar chunking 30s estilo Whisper.
- Generar mel-spectrogram 80 mels.
- Tokenizar con BPE Whisper.
- Diagnosticar Whisper vs faster-whisper vs Distil.

## Constrúyelo

```python
def chunk_audio(senal, max_seconds=30, sample_rate=16000):
    max_samples = max_seconds * sample_rate
    n_chunks = (len(senal) + max_samples - 1) // max_samples
    chunks = []
    for i in range(n_chunks):
        chunk = senal[i * max_samples:(i + 1) * max_samples]
        if len(chunk) < max_samples:
            chunk = np.pad(chunk, (0, max_samples - len(chunk)))
        chunks.append(chunk)
    return chunks
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
fase: 06
leccion: 05
---

1. Produccion: faster-whisper (4x speedup).
2. API: Whisper-large-v3, GPT-4o-transcribe.
3. Self-host: Distil-Whisper (edge).
4. Multilingual: Whisper v3 (99 idiomas).
5. Fine-tune: Common Voice + LoRA.
6. 16kHz mono, 30s chunks, 80 mels log.
```

## Ejercicios

1. **faster-whisper**: usar faster-whisper para
   transcribir un audio espanol.
2. **Fine-tune**: fine-tunear Whisper-small en
   Common Voice espanol subset.
3. **Desafio**: pipeline production con
   faster-whisper + VAD + punctuation + ITN + diarization,
   eval WER en golden set.

## Lecturas recomendadas

- "Whisper" (Radford et al., 2022)
- "Distil-Whisper" (HuggingFace, 2023)
- faster-whisper: <https://github.com/SYSTRAN/faster-whisper>

---

> 📚 **Adaptación al español** de la lección "[Whisper Architecture and Fine-tuning]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).