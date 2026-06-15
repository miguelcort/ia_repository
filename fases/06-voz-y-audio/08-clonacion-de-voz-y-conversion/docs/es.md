# 08 — Clonación de voz y conversión

> Voice cloning: dado unos segundos de audio de referencia, generar habla con esa voz. Voice conversion: cambiar la voz de un audio a otra. SOTA con XTTS, So-VITS-SVC, RVC.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07-texto-a-habla-tts
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar zero-shot voice cloning con XTTS.
- Aplicar voice conversion con RVC o So-VITS-SVC.
- Diagnosticar la calidad y la similitud de voz.
- Conocer las implicaciones éticas y legales.

## El problema

Voice cloning genera habla en la voz de un hablante de
referencia con solo unos segundos de audio. Voice
conversion transforma la voz en un audio existente (e.g.
tu voz a la de tu celebridad favorita). Ambos son
tecnologías poderosas con aplicaciones legítimas
(doblaje, accesibilidad, contenido personalizado) y
riesgos (deepfakes, fraude, desinformación).

## El concepto

**Voice cloning.**

- **Few-shot / zero-shot:** XTTS (Coqui, 2024) clona
  una voz con solo 6 segundos de audio. Multilingual
  (~17 idiomas).
- **One-shot:** models como Vall-E o NaturalSpeech 3
  (Microsoft) clonan con 3 segundos.
- **Many-shot:** fine-tuning completo con varios minutos
  de audio. Calidad SOTA, pero requiere recursos.

**Voice conversion.**

- **RVC (Retrieval-based Voice Conversion):** el más
  popular en 2024-2025. Few-shot, alta calidad.
  Project + index + inference.
- **So-VITS-SVC:** basado en VITS. Soft-VC, soft
  content features. Muy buena calidad.
- **Diff-VC:** usa difusión para voice conversion.
  SOTA en naturalidad.
- **QuickVC / kNN-VC:** zero-shot con kNN.

**Pipeline típico (RVC).**

1. Pre-procesar audio de target (canto, voz) con
   `preprocess.py` (separar vocals, segmentar, extraer
   pitch con RMVPE).
2. Entrenar modelo RVC con el audio procesado.
3. Inferencia: tomar audio de source, extraer pitch y
   content, generar con el modelo target.
4. Post-procesar: aplicar vocoder (HiFi-GAN) y
   opcionalmente un upsampler.

**Métricas.**

- **MOS:** naturalidad, evaluada por humanos.
- **SIM:** similitud con el hablante target.
- **WER con ASR:** intelligibilidad.
- **Naturalidad:** que no suene robótico.

**Aplicaciones legítimas.**

- **Doblaje:** clonar la voz del actor original a otro
  idioma.
- **Accesibilidad:** personas que pierden la voz
  pueden clonarla antes de perderla.
- **Contenido:** podcasts, audiolibros personalizados.
- **Videojuegos:** NPCs con voces dinámicas.

**Riesgos y mitigaciones.**

- **Deepfakes:** fraude (CEO que simula ser
  ejecutivo), desinformación.
- **Consentimiento:** no clonar la voz de alguien sin
  permiso explícito.
- **Detección:** AudioSeal, RawNet, otros detectores
  de audio sintético.
- **Watermarking:** marcas de agua invisibles en
  audio generado (SynthID).

**Trampas.**

- **Calidad de audio de referencia:** 6s de audio
  limpio vs 6s de audio con ruido. XTTS es muy sensible
  a la calidad.
- **Sobreajuste a una voz:** few-shot puede
  sobreajustarse a la referencia. Validar con audios
  de test no vistos.
- **Idioma no soportado:** XTTS soporta 17 idiomas
  pero la calidad varía. Español está bien; aimara no.

## Constrúyelo

```python
import numpy as np


def extract_voice_embedding(audio, sr=16000):
    """Embedding de hablante (x-vector simplificado).
    En producción: SpeechBrain ECAPA-TDNN."""
    # Placeholder: media del waveform
    return audio.mean(axis=-1) if audio.ndim > 1 else audio


def voice_clone_tts(text, reference_audio, sr=24000):
    """TTS con voice cloning. En producción: XTTS, Vall-E."""
    # Placeholder: retorna waveform aleatorio con misma duración
    duration = len(reference_audio) / sr
    n_samples = int(duration * sr)
    return np.random.randn(n_samples).astype(np.float32) * 0.1


def voice_conversion(source_audio, target_embedding, sr=16000):
    """Voice conversion: cambiar la voz del source a la del
    target embedding. En producción: RVC, So-VITS-SVC."""
    # Placeholder: retorna source con modifications
    return source_audio * 0.9
```

## Úsalo

```bash
pip install TTS
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-voice-cloning
fase: 06
leccion: 08
---

Eres un asistente que ayuda con voice cloning o con-
versión. Recibirás el audio de referencia y la aplica-
ción. Tu trabajo:

1. Si quieres zero-shot: XTTS con 6s de referencia.
2. Si quieres voice conversion: RVC o So-VITS-SVC.
3. Pre-procesar: separar vocals, segmentar, extraer
   pitch.
4. Evaluar MOS, SIM, WER con ASR.
5. SIEMPRE obtener consentimiento explícito del
   hablante original.
6. Watermark: SynthID, AudioSeal.
7. Anti-spoofing: detector de audio sintético.
8. Advertir contra uso malicioso (fraude, deepfakes).
9. Logging: cada generación con timestamp y referencia
   al audio original.
```

## Ejercicios

1. **XTTS**: clona tu voz con 6s de audio.
2. **RVC**: convierte tu voz a la de un cantante.
3. **Desafío**: implementa un detector de audio
   sintético (anti-spoofing).

## Lecturas recomendadas

- *XTTS* — Coqui, 2024.
- *RVC* — <https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI>.
- *Vall-E* — Wang et al., 2023.
- *NaturalSpeech 3* — Microsoft, 2024.
- AudioSeal: <https://github.com/facebookresearch/audioseal>.

---

> 📚 **Adaptación al español** de la lección "[Voice Cloning and Conversion]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
