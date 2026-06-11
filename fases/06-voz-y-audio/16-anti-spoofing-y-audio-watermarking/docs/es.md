# Anti-spoofing y audio watermarking

> Detectar audio sintetico (RawNet, AASIST2) y marcarlo con watermark (AudioSeal, SilentCipher). Critico para: voice biometrics, EU AI Act (AI labeling), copyright, anti-deepfake.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 15-streaming-speech-to-speech
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar detector de deepfake mock.
- Implementar anti-spoofing real-time mock.
- Implementar audio watermarking y deteccion.
- Diagnosticar compliance (EU AI Act, China).

## Constrúyelo

```python
def embed_watermark(audio, message="CREATED BY AI"):
    rng = np.random.default_rng(hash(message) % 2**32)
    watermark = 0.001 * rng.choice([-1, 1], size=len(audio))
    return audio + watermark
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-anti-spoofing
fase: 06
leccion: 16
---

1. Default: AudioSeal + RawNet/AASIST2.
2. Voice biometrics: AASIST2 (SOTA ASVspoof 5).
3. AI content labeling: AudioSeal + Resemble Detect.
4. Compliance: EU AI Act, China AI labeling.
5. Frameworks: AudioSeal, RawNet, AASIST.
6. Watermarking 16-32 bps imperceptible; audit logs.
```

## Ejercicios

1. **AudioSeal watermark**: insertar watermark en audio
   y verificarlo.
2. **RawNet detection**: usar RawNet para detectar
   audio sintetico.
3. **Desafio**: pipeline production con AudioSeal en
   TTS output + RawNet en ingestion + logging para
   compliance (EU AI Act).

## Lecturas recomendadas

- "AudioSeal" (Meta, 2024)
- "RawNet" (Jung et al., 2020)
- "AASIST" (Jung et al., 2021)
- EU AI Act: <https://artificialintelligenceact.eu/>

---

> 📚 **Adaptación al español** de la lección "[Anti-Spoofing and Audio Watermarking]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).