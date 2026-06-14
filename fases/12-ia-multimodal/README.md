# Fase 12 — IA multimodal

> Ver, oír, leer y razonar a través de modalidades.

Los **modelos multimodales** son el siguiente paso de los LLMs: en
lugar de procesar sólo texto, entienden imágenes (LLaVA, Qwen-VL,
InternVL), audio (Whisper, Qwen-Audio), video (VideoLLaMA,
LLaVA-NeXT-Video), documentos (ColPali, Qwen-VL-OCR), y
combinaciones de todos ellos (GPT-4o, Gemini 1.5 Pro, MIO). Esta
fase recorre la **arquitectura multimodal moderna**: cómo se
codifican las modalidades, cómo se alinean en un espacio común, cómo
se hace *instruction tuning* visual, y cómo se entrenan modelos
generativos *any-to-any*.

La fase se organiza en **cuatro bloques**. El **bloque 1**
(lecciones 1–5) cubre la **primitiva patch-token y la familia
CLIP/LLaVA**: ViT, CLIP, BLIP-2, Flamingo, LLaVA. El **bloque 2**
(6–10) trata **VLMs de cualquier resolución y recetas modernas**:
Patch-n-Pack, NaFlex, LLaVA-OneVision, Qwen-VL, InternVL3. El
**bloque 3** (11–16) entra en los **modelos generativos
any-to-any**: Chameleon, Emu3, Transfusion, Show-o, Janus-Pro, MIO.
El **bloque 4** (17–25) cubre el **largo plazo**: video
temporal, contexto de millón de tokens, audio-lenguaje, omni
models, *vision-language-action* (VLA), RAG multimodal y agentes
multimodales.

## Índice de lecciones

### Bloque 1 — Primitiva patch-token y la familia CLIP/LLaVA

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [Vision Transformer y la primitiva patch-token](01-vision-transformer-y-patch-tokens/) | Aprender | ViT, *patch embeddings* y tokens visuales. |
| 02 | [CLIP y pre-entrenamiento contrastivo](02-clip-contrastive-pretraining/) | Construir | Contrastive image-text, InfoNCE. |
| 03 | [BLIP-2 y Q-Former como puente](03-blip2-y-qformer-bridge/) | Construir | Q-Former, ITC, ITM, ITG. |
| 04 | [Flamingo y cross-attention con gate](04-flamingo-gated-cross-attention/) | Aprender | *Gated cross-attention* para few-shot visual. |
| 05 | [LLaVA y visual instruction tuning](05-llava-visual-instruction-tuning/) | Construir | MLP projector + LLM + visual SFT. |

### Bloque 2 — VLMs modernos y any-resolution

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 06 | [Visión any-resolution: Patch-n'-Pack y NaFlex](06-any-resolution-patch-n-pack/) | Construir | Tiling dinámico y *aspect ratio* flexible. |
| 07 | [Recetas de VLMs open-weight](07-open-weight-vlm-recipes/) | Aprender | LLaVA, Idefics, MiniCPM-V, Pixtral. |
| 08 | [LLaVA-OneVision: single, multi, video](08-llava-onevision-single-multi-video/) | Construir | *Single-image*, *multi-image* y *video* unificados. |
| 09 | [Familia Qwen-VL y video a FPS dinámico](09-qwen-vl-family-dynamic-fps/) | Aprender | Qwen2-VL, Qwen2.5-VL, *dynamic resolution*. |
| 10 | [InternVL3 — multimodal nativo](10-internvl3-native-multimodal/) | Aprender | *Native multimodal pretraining*. |

### Bloque 3 — Modelos generativos any-to-any

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 11 | [Chameleon: early-fusion solo tokens](11-chameleon-early-fusion-tokens/) | Construir | *Early fusion* con tokenizer de imagen. |
| 12 | [Emu3: next-token para generación](12-emu3-next-token-for-generation/) | Aprender | Todo como *next token prediction*. |
| 13 | [Transfusion: autoregresivo + difusión](13-transfusion-autoregressive-diffusion/) | Construir | Texto AR + imagen con difusión. |
| 14 | [Show-o: discrete-diffusion unificado](14-show-o-discrete-diffusion-unified/) | Aprender | Discrete diffusion cross-modal. |
| 15 | [Janus-Pro: encoders desacoplados](15-janus-pro-decoupled-encoders/) | Construir | *Dual-encoder* y *decoupled* visual encoding. |
| 16 | [MIO: any-to-any streaming](16-mio-any-to-any-streaming/) | Aprender | *Streaming* multimodal a baja latencia. |

### Bloque 4 — Largo plazo: video, audio, omni, embodied

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 17 | [Grounding temporal video-lenguaje](17-video-language-temporal-grounding/) | Construir | Localización temporal de eventos. |
| 18 | [Video largo a contexto de millón de tokens](18-long-video-million-token/) | Construir | Compresión temporal, *memory tokens*. |
| 19 | [Modelos audio-lenguaje](19-audio-language-whisper-to-af3/) | Construir | Whisper, Qwen-Audio, Audio Flamingo. |
| 20 | [Modelos omni: thinker-talker](20-omni-models-thinker-talker/) | Construir | Arquitectura *dual-channel* para voz en tiempo real. |
| 21 | [VLAs embodied: RT-2, OpenVLA, π0, GR00T](21-embodied-vlas-openvla-pi0-groot/) | Aprender | *Vision-language-action* para robots. |
| 22 | [Comprensión de documentos y diagramas](22-document-diagram-understanding/) | Construir | Layout, OCR, tablas, figuras. |
| 23 | [ColPali: RAG visión-nativa](23-colpali-vision-native-rag/) | Construir | *Vision encoder* sobre páginas. |
| 24 | [RAG multimodal y retrieval cross-modal](24-multimodal-rag-cross-modal/) | Construir | Texto-imagen-audio en el mismo índice. |
| 25 | [Agentes multimodales y computer-use (capstone)](25-multimodal-agents-computer-use/) | Construir | Captura de pantalla + acciones. |

## Prerrequisitos

- **Fases 4, 5, 7, 10 y 11** completas.
- Conocimiento sólido de ViT, CLIP y LLaVA.
- GPU con 24+ GB VRAM para las lecciones prácticas.

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Explicar** la primitiva *patch-token* y por qué reemplazó a
  las CNNs en la mayoría de los VLMs modernos.
- **Implementar** CLIP desde cero y pre-entrenarlo en un dataset
  pequeño.
- **Construir** un LLaVA mínimo con ViT + projector + LLM y
  hacer *visual instruction tuning*.
- **Comparar** arquitecturas any-to-any: Chameleon, Emu3,
  Transfusion, Show-o, Janus-Pro.
- **Entrenar** un VLA para un robot simulado.
- **Construir** un RAG multimodal con ColPali o similar.
- **Desplegar** un agente multimodal con capacidad de computer
  use (Claude, GPT-4o, Gemini).

## Stack y herramientas

- **transformers** y **accelerate** de Hugging Face.
- **LLaVA, Idefics, Qwen-VL, InternVL, Janus-Pro** como modelos
  base.
- **OpenCLIP** para CLIP reproducible.
- **ColPali** para RAG visión-nativa.
- **vLLM** para servir VLMs.
- **trl, axolotl, LLaMA-Factory** para SFT multimodal.
- **RT-2, OpenVLA, π0, GR00T** para VLA.
- **OpenCV, Pillow, decord** para I/O de video.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **Patch-token** | Lección 01 | Cualquier VLM. |
| **CLIP** | Lección 02 | Fase 4 (imagen), Fase 14 (búsqueda). |
| **Q-Former** | Lección 03 | BLIP-2. |
| **Visual instruction tuning** | Lección 05 | LLaVA, Idefics. |
| **Any-resolution** | Lección 06 | NaFlex, dynamic resolution. |
| **Early fusion** | Lección 11 | Chameleon, Emu3. |
| **VLA** | Lección 21 | Robótica. |
| **RAG multimodal** | Lecciones 23, 24 | Agentes. |

## Cómo estudiar esta fase

1. **Empieza por la lección 01 (ViT).** Sin entender
   *patch-tokens*, las lecciones 02–10 parecen magia.
2. **CLIP es la lección más importante.** Todo lo demás se basa
   en contrastive learning o en投影 lineal del espacio CLIP.
3. **No te abrumes con todas las arquitecturas any-to-any.**
   Elige una (Chameleon, Emu3 o Transfusion) y entiéndela a
   fondo; las demás son variaciones.
4. **Las lecciones 23 y 25 son prácticas y valiosas.** Si vas
   corto de tiempo, priorízalas.
5. **VLA (lección 21) es la frontera.** Si te interesa la
   robótica, ahonda ahí; si no, sáltala.

## Verificación de progreso

```bash
# Lección 02 — CLIP contrastive
python3 fases/12-ia-multimodal/02-clip-contrastive-pretraining/code/main.py

# Lección 05 — LLaVA minimal
python3 fases/12-ia-multimodal/05-llava-visual-instruction-tuning/code/main.py

# Lección 23 — ColPali RAG
python3 fases/12-ia-multimodal/23-colpali-vision-native-rag/code/main.py
```

Si los tres demos terminan con código 0, la fase está aprobada.

## Cuándo usar cada familia

| Caso de uso | Familia | Lección |
|---|---|---|
| Captioning / VQA | LLaVA, Qwen-VL | 05, 09 |
| Búsqueda imagen↔texto | CLIP | 02 |
| Documentos / OCR | ColPali, Qwen-VL-OCR | 22, 23 |
| Generación de imagen | Emu3, Transfusion, Janus | 12, 13, 15 |
| Video understanding | LLaVA-OneVision, Qwen-VL | 08, 09 |
| RAG multimodal | ColPali, AnyModal | 23, 24 |
| Robotics / embodied | VLA (π0, OpenVLA) | 21 |
| Agente con pantalla | Claude, GPT-4o, Qwen-VL | 25 |

## Conexión con otras fases

- **Entrada** → [Fase 4 — Visión](../04-vision-por-computador/README.md),
  [Fase 5 — NLP](../05-nlp-fundamentos-a-avanzado/README.md) y
  [Fase 11 — Ingeniería de LLMs](../11-ingenieria-llms/README.md).
- **Salida natural** → [Fase 14 — Ingeniería de agentes](../14-ingenieria-agentes/README.md)
  (los agentes multimodales son la siguiente capa) y
  [Fase 15 — Sistemas autónomos](../15-sistemas-autonomos/README.md).
- **Reuso en** → Fase 19 (capstone).

## Recursos recomendados

- *CLIP paper* — Radford et al., 2021.
- *LLaVA paper* — Liu et al., 2023.
- *Qwen-VL paper* — Bai et al., 2023.
- *Show-o paper* — Xie et al., 2024.
- *Transfusion paper* — Zhou et al., 2024.
- *OpenVLA paper* — Kim et al., 2024.
- *π0 paper* — Black et al., 2024.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *ViT*,
  *CLIP*, *VLM*, *VLA*, *multimodal RAG*.
- [Fase 4 — Visión](../04-vision-por-computador/README.md).
- [Fase 14 — Ingeniería de agentes](../14-ingenieria-agentes/README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
