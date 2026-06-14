# Fase 6 — Voz y audio

> Escuchar, entender, hablar.

La voz y el audio son la **interfaz más natural** entre humanos y
máquinas. Un asistente que escucha y habla se siente menos como una
herramienta y más como un colaborador. Esta fase cubre todo el
ciclo: capturar audio del micrófono, entender el habla (ASR),
responder en texto, y volver a hablar (TTS). También aborda
reconocimiento de hablante, generación de música, modelos
audio-lenguaje, codecs neuronales, anti-spoofing y métricas de
evaluación.

El recorrido está organizado en **cuatro bloques**. El **bloque 1**
(lecciones 1–3) sienta las bases: waveforms, espectrogramas, escala
Mel y clasificación de audio. El **bloque 2** (4–6) cubre la
**comprensión del habla**: ASR clásico, Whisper y variantes, y
reconocimiento de hablante. El **bloque 3** (7–12) entra en la
**generación de audio**: TTS, clonación de voz, generación de
música, modelos audio-lenguaje, audio en tiempo real y el capstone
de asistente de voz. El **bloque 4** (13–17) cubre la **ingeniería
del audio**: codecs neuronales, VAD, streaming speech-to-speech,
anti-spoofing y métricas.

## Índice de lecciones

### Bloque 1 — Fundamentos

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [Fundamentos de audio](01-fundamentos-de-audio/) | Aprender | Waveforms, frecuencia de muestreo, FFT, STFT. |
| 02 | [Espectrogramas y características Mel](02-espectrogramas-y-caracteristicas-mel/) | Construir | STFT, banco de filtros Mel, MFCC. |
| 03 | [Clasificación de audio](03-clasificacion-de-audio/) | Construir | PANN, AST, ESC-50, UrbanSound8K. |

### Bloque 2 — Comprensión del habla

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 04 | [Reconocimiento de voz (ASR)](04-reconocimiento-de-habla-asr/) | Construir | CTC, *language model*, KenLM y WER. |
| 05 | [Whisper: arquitectura y fine-tuning](05-arquitectura-whisper-y-fine-tuning/) | Construir | Encoder-decoder, timestamp tokens, *fine-tune* LoRA. |
| 06 | [Reconocimiento de hablante y verificación](06-reconocimiento-de-hablante-y-verificacion/) | Construir | d-vector, x-vector, ECAPA-TDNN. |

### Bloque 3 — Generación y asistentes

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 07 | [Texto a voz (TTS)](07-texto-a-habla-tts/) | Construir | Tacotron, FastSpeech, VITS, Bark. |
| 08 | [Clonación de voz y conversión](08-clonacion-de-voz-y-conversion/) | Construir | Tortoise, So-VITS-SVC, RVC, XTTS. |
| 09 | [Generación de música](09-generacion-de-musica/) | Construir | MusicGen, AudioLDM, Stable Audio. |
| 10 | [Modelos audio-lenguaje](10-modelos-de-lenguaje-de-audio/) | Construir | Qwen-Audio, SALMONN, GAMA. |
| 11 | [Procesamiento de audio en tiempo real](11-procesamiento-de-audio-en-tiempo-real/) | Construir | WebRTC, VAD, *streaming*, *circular buffers*. |
| 12 | [Pipeline de asistente de voz (capstone)](12-pipeline-de-asistente-de-voz/) | Construir | STT → LLM → TTS en una sola aplicación. |

### Bloque 4 — Ingeniería de audio

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 13 | [Codecs neuronales de audio](13-codecs-de-audio-neurales/) | Aprender | EnCodec, SNAC, Mimi, DAC. |
| 14 | [Voice activity detection y turn-taking](14-deteccion-de-actividad-de-voz-y-turn-taking/) | Construir | Silero VAD, pyannote, *end-of-turn*. |
| 15 | [Streaming speech-to-speech](15-streaming-speech-to-speech/) | Aprender | Moshi, Hibiki, paralelismo audio-tokens. |
| 16 | [Anti-spoofing y marcas de agua](16-anti-spoofing-y-audio-watermarking/) | Construir | RawNet, AudioSeal, SynthID. |
| 17 | [Métricas de evaluación de audio](17-metricas-de-evaluacion-de-audio/) | Aprender | WER, MOS, FAD, MMAU, leaderboards. |

## Prerrequisitos

- **Fases 0, 1, 3 y 4** completas.
- **Fase 1** recomendada: la lección 20 (transformada de Fourier)
  es la base del análisis espectral.
- Conocimiento de PyTorch.
- GPU recomendada para *fine-tuning*; ASR y TTS *inferencia*
  funcionan en CPU para casos de uso modestos.

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Manipular** audio digital: muestreo, cuantización, filtros,
  espectrogramas y características Mel.
- **Construir** un pipeline ASR con Whisper o un modelo CTC.
- **Generar** habla con TTS moderno (VITS, Bark, XTTS).
- **Clonar** una voz con técnicas de *few-shot voice cloning*.
- **Construir** un asistente de voz funcional en tiempo real
  (STT → LLM → TTS).
- **Evaluar** audio con WER, MOS, FAD, MMAU y otros benchmarks.
- **Detectar** audio generado por IA y aplicar marcas de agua.

## Stack y herramientas

- **librosa**, **torchaudio**, **soundfile** para I/O y DSP.
- **transformers** para Whisper.
- **pyannote-audio** para diarización y VAD.
- **TTS** (Coqui) y **Bark** para TTS.
- **EnCodec, SNAC, Mimi** para codecs neuronales.
- **WebRTC** y **Pipecat** para streaming.
- **AudioSeal** y **SynthID** para watermarking.
- **Pyaudio** para captura desde micrófono.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **Espectrograma** | Lección 01, 02 | Cualquier tarea de audio. |
| **Mel-scale** | Lección 02 | ASR, TTS, music. |
| **CTC** | Lección 04 | Reconocimiento secuencial. |
| **Whisper** | Lección 05 | Asistencia universal. |
| **d-vector / x-vector** | Lección 06 | Verificación de hablante. |
| **VITS** | Lección 07 | TTS moderno. |
| **Codec neural** | Lección 13 | Streaming, TTS, S2S. |
| **VAD** | Lección 14 | Turn-taking. |
| **Watermarking** | Lección 16 | Ética, anti-spoofing. |

## Cómo estudiar esta fase

1. **La lección 02 (espectrogramas) es el cuello de botella.** Sin
   entender STFT y Mel, el resto parece magia.
2. **Empieza con Whisper antes que con Wav2Vec2.** Whisper es
   end-to-end y mucho más fácil de usar.
3. **Las lecciones 11 y 12 son prácticas y valiosas.** Si vas
   corto de tiempo, priorízalas.
4. **Para TTS, VITS es el mejor equilibrio calidad/facilidad.**
   Bark da mejor calidad pero requiere más GPU.
5. **Las métricas importan.** WER para ASR, MOS para TTS
   subjetivo, FAD para generación de música.

## Verificación de progreso

```bash
# Lección 02 — espectrograma Mel
python3 fases/06-voz-y-audio/02-espectrogramas-y-caracteristicas-mel/code/main.py

# Lección 05 — Whisper fine-tune
python3 fases/06-voz-y-audio/05-arquitectura-whisper-y-fine-tuning/code/main.py

# Lección 12 — asistente de voz
python3 fases/06-voz-y-audio/12-pipeline-de-asistente-de-voz/code/main.py
```

Si los tres demos terminan con código 0, la fase está aprobada.

## Cuándo usar cada técnica

| Objetivo | Técnica | Lección |
|---|---|---|
| Transcribir audio | Whisper, CTC | 04, 05 |
| Identificar hablante | ECAPA-TDNN | 06 |
| Generar habla desde texto | VITS, Bark, XTTS | 07, 08 |
| Generar música | MusicGen, Stable Audio | 09 |
| Asistente interactivo | Whisper + LLM + TTS | 11, 12 |
| Detectar deepfake de audio | RawNet, AudioSeal | 16 |
| Medir calidad TTS | MOS, FAD | 17 |

## Conexión con otras fases

- **Entrada** → [Fase 4 — Visión](../04-vision-por-computador/README.md)
  y [Fase 5 — NLP](../05-nlp-fundamentos-a-avanzado/README.md).
- **Salida natural** → [Fase 14 — Ingeniería de agentes](../14-ingenieria-agentes/README.md)
  (los voice agents son un caso de uso de agentes).
- **Reuso en** → Fase 10 (LLM), Fase 12 (multimodal), Fase 19 (capstone).

## Recursos recomendados

- *Speech and Language Processing* — Jurafsky & Martin (cap. sobre habla).
- *Fundamentals of Speech Recognition* — Rabiner, Juang.
- *Whisper paper* — Radford et al., 2022.
- *Hugging Face Audio course* — <https://huggingface.co/learn/audio-course>.
- *TTS papers* — Tacotron 2, FastSpeech 2, VITS.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *ASR*, *TTS*,
  *codec*, *MOS*, *WER*.
- [Fase 5 — NLP](../05-nlp-fundamentos-a-avanzado/README.md).
- [Fase 14 — Ingeniería de agentes](../14-ingenieria-agentes/README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
