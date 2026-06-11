# Métricas de evaluación de audio

> PESQ (perceptual), STOI (intelligibility), MOS (1-5 human), FAD (Frechet Audio Distance), DNSMOS (no-reference), WER-ASR (TTS intelligibility), speaker similarity. Frameworks: torchmetrics.audio, ESPnet-SQUIM, speech-mos, PESQ lib.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16-anti-spoofing-y-audio-watermarking
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar PESQ mock.
- Implementar STOI y ViSQOL mock.
- Calcular MOS estimado.
- Calcular FAD entre embeddings.
- Diagnosticar cuando usar cada metrica.

## Constrúyelo

```python
def pesq_score(ref, deg, sample_rate=16000):
    if len(ref) != len(deg):
        n = min(len(ref), len(deg))
        ref = ref[:n]; deg = deg[:n]
    if np.std(ref) == 0 or np.std(deg) == 0: return 1.0
    corr = float(np.corrcoef(ref, deg)[0, 1])
    return max(1.0, min(4.5, 2.0 + 2 * corr))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-audio-eval
fase: 06
leccion: 17
---

1. TTS: MOS, WER-ASR, speaker similarity, MCD, PESQ.
2. Music: FAD, KLD, CLAP score, MOS.
3. Audio: FAD, CLAP, MOS, DNSMOS.
4. Real-time: PESQ, STOI, DNSMOS.
5. Frameworks: torchmetrics.audio, ESPnet-SQUIM, W&B.
6. A/B testing, drift monitoring, golden set.
```

## Ejercicios

1. **PESQ real**: usar pesq lib para calcular PESQ en
   audios reales.
2. **FAD**: calcular FAD entre audios de MusicGen y
   reales.
3. **Desafio**: framework de eval production con golden
   set 100 audios, MOS human eval, PESQ, FAD, WER-ASR.
   A/B test entre 2 modelos TTS.

## Lecturas recomendadas

- "PESQ" (Rix et al., 2001)
- "FAD" (Kilgour et al., 2019)
- torchmetrics.audio: <https://torchmetrics.readthedocs.io/>
- ESPnet-SQUIM: <https://github.com/espnet/espnet>

---

> 📚 **Adaptación al español** de la lección "[Audio Evaluation Metrics]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).