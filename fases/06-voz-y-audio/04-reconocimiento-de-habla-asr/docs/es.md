# Reconocimiento de habla (ASR)

> Speech -> text. End-to-end con Whisper, wav2vec 2.0, Conformer, HuBERT, WavLM, Canary. WER 5-10% SOTA en benchmarks limpios, 15-30% en noisy. Aplicaciones: subtitulos, voice assistants, transcripcion, call center.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-clasificacion-de-audio
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Calcular WER y CER.
- Implementar mock ASR decoder.
- Diagnosticar Whisper vs wav2vec vs HuBERT.
- Evaluar con benchmarks publicos.

## Constrúyelo

```python
def wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n] / m if m > 0 else 0.0
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-asr
fase: 06
leccion: 04
---

1. Default: Whisper-large-v3 (99 idiomas).
2. Self-host: wav2vec 2.0, Canary, Nemo.
3. Multilingual: Whisper, SeamlessM4T, MMS.
4. Streaming: faster-whisper, Vosk.
5. WER, CER, RTF; punctuation, ITN, diarization.
```

## Ejercicios

1. **faster-whisper**: usar faster-whisper para
   transcribir un audio.
2. **VAD + ASR**: integrar silero VAD con Whisper.
3. **Desafio**: ASR pipeline production con
   faster-whisper, VAD, punctuation, ITN, diarization.
   Eval WER en Common Voice espanol.

## Lecturas recomendadas

- "Whisper" (Radford et al., 2022)
- "wav2vec 2.0" (Baevski et al., 2020)
- faster-whisper: <https://github.com/SYSTRAN/faster-whisper>

---

> 📚 **Adaptación al español** de la lección "[ASR]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).