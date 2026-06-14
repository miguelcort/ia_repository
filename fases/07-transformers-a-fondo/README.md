# Fase 7 — Transformers a fondo

> La arquitectura que lo cambió todo.

El **transformer** es la arquitectura central de la IA moderna.
Dominó el lenguaje (BERT, GPT, T5, LLaMA, Mistral), cruzó a visión
(ViT, SAM, DINO), audio (Whisper, Wav2Vec2), y multimodales (CLIP,
LLaVA, Flamingo). Entender el transformer en detalle es **no
negociable** para cualquier ingeniero de IA en 2026: cada decisión
de modelado, *fine-tuning*, despliegue y optimización presupone
que sabes qué hay dentro de esos 200-3000 millones de parámetros.

La fase se organiza en **cuatro bloques**. El **bloque 1** (1–5)
cubre los componentes básicos: por qué los transformers reemplazaron
a las RNN, self-attention desde cero, multi-head, codificación
posicional y el bloque transformer completo. El **bloque 2** (6–10)
presenta las **tres familias** de modelos preentrenados: encoder-
only (BERT), decoder-only (GPT) y encoder-decoder (T5, BART), más
las variantes multimodales (ViT, Whisper). El **bloque 3** (11–13)
escala hacia la producción: Mixture of Experts, KV cache + Flash
Attention y las leyes de escalado de Kaplan/ Hoffmann. El **bloque
4** (14–16) cierra con un capstone: construir un transformer desde
cero, variantes de atención eficientes y speculative decoding.

## Índice de lecciones

### Bloque 1 — Componentes del transformer

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [Por qué Transformers](01-por-que-transformers/) | Aprender | Las limitaciones de las RNN y el papel de la atención. |
| 02 | [Self-attention desde cero](02-self-attention-desde-cero/) | Construir | Scaled dot-product attention implementada a mano. |
| 03 | [Multi-head attention](03-multi-head-attention/) | Construir | Múltiples sub-espacios, paralelo y concat final. |
| 04 | [Codificación posicional](04-positional-encoding/) | Construir | Sinusoidal, aprendida, RoPE y ALiBi. |
| 05 | [Transformer completo](05-transformer-completo/) | Construir | Encoder + decoder, *layer norm* y *residual*. |

### Bloque 2 — Familias de modelos

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 06 | [BERT — masked language modeling](06-bert-masked-language-modeling/) | Construir | Encoder-only, MLM, NSP y *fine-tuning*. |
| 07 | [GPT — causal language modeling](07-gpt-causal-language-modeling/) | Construir | Decoder-only, *next token prediction*. |
| 08 | [T5, BART — encoder-decoder](08-t5-bart-encoder-decoder/) | Aprender | *Span corruption*, *denoising*, *unified* text-to-text. |
| 09 | [Vision Transformers (ViT)](09-vision-transformers/) | Construir | *Patches*, *positional embeddings*, atención axial. |
| 10 | [Audio Transformers — Whisper](10-audio-transformers-whisper/) | Aprender | *Log-mel*, encoder-decoder, *multitask*. |

### Bloque 3 — Producción y escala

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 11 | [Mixture of Experts (MoE)](11-mixture-of-experts/) | Construir | *Top-k routing*, *load balancing*, *sparse* FFN. |
| 12 | [KV cache y Flash Attention](12-kv-cache-y-flash-attention/) | Construir | Inferencia eficiente, *tiling*, *memory savings*. |
| 13 | [Leyes de escalado](13-leyes-de-escalado/) | Aprender | Kaplan/Hoffmann, *compute optimal*, *emergent* abilities. |

### Bloque 4 — Capstone y optimizaciones

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 14 | [Construye un Transformer desde cero](14-construye-un-transformer-capstone/) | Construir | 200 líneas de PyTorch entrenadas en TinyStories. |
| 15 | [Variantes de atención](15-variantes-de-atencion/) | Construir | Sliding window, sparse, linear, differential. |
| 16 | [Speculative decoding](16-speculative-decoding/) | Construir | Draft, verify, accept/reject sampling. |

## Prerrequisitos

- **Fases 0, 1, 3, 4 y 5** completas.
- Conocimiento sólido de PyTorch y de las redes neuronales básicas
  (Fase 3).
- Recomendada: la **Fase 5 (NLP)** por familiaridad con tareas de
  lenguaje.
- GPU recomendada (16+ GB VRAM) para las lecciones 11, 12 y 14.

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Derivar** la fórmula de scaled dot-product attention e
  implementarla en NumPy.
- **Construir** un bloque transformer completo (multi-head +
  FFN + residuals + layer norm) en PyTorch.
- **Comparar** las tres familias (encoder, decoder, encoder-
  decoder) y elegir la arquitectura adecuada para una tarea.
- **Entrenar** un transformer pequeño en un dataset modesto
  (TinyStories, Shakespeare).
- **Optimizar** inferencia con KV cache, Flash Attention y
  speculative decoding.
- **Leer** papers de 2024-2026 sobre LLMs y entender la
  terminología (*RoPE*, *GQA*, *MoE*, *sliding window*).
- **Razonar** sobre el *trade-off* parámetros vs. *tokens* vs.
  cómputo con las leyes de escalado.

## Stack y herramientas

- **PyTorch** (≥ 2.2) con `torch.compile` y `torch.nn.functional.scaled_dot_product_attention`.
- **transformers** y **datasets** de Hugging Face.
- **flash-attn** opcional para lecciones 12 y 15.
- **Tiktoken** o **tokenizers** para la lección 14.
- **wandb** para tracking.
- **Einops** para manipulación elegante de tensores.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **Self-attention** | Lección 02 | Cualquier transformer. |
| **Multi-head** | Lección 03 | Fase 4 (ViT), Fase 6 (Whisper). |
| **RoPE** | Lección 04 | LLaMA, Mistral, Qwen. |
| **Layer norm** | Lección 05 | Cualquier red profunda. |
| **KV cache** | Lección 12 | Inferencia de LLM. |
| **Flash Attention** | Lección 12 | Todas las GPUs modernas. |
| **MoE** | Lección 11 | Mixtral, DeepSeek-V3. |
| **Speculative decoding** | Lección 16 | Inferencia rápida. |
| **Sliding window** | Lección 15 | Mistral, Long context. |

## Cómo estudiar esta fase

1. **No te saltes las lecciones 2-5.** Son la columna vertebral
   de todo lo demás. Dedica al menos un día entero a cada una.
2. **Compara siempre tu implementación con `torch.nn.MultiheadAttention`.**
   Si los pesos convergen al mismo resultado, vas bien.
3. **La lección 14 (capstone) es el examen real.** Si puedes
   entrenar un transformer en TinyStories y generar texto
   coherente, dominas la fase.
4. **Las lecciones 12 y 16 son las más valiosas para
   ingeniería de LLM en 2026.** Si tu tiempo es limitado,
   priorízalas.
5. **Leyes de escalado (13) es más conceptual.** Léela con un
   café y un cuaderno; no hay que programar nada.

## Verificación de progreso

```bash
# Lección 02 — self-attention desde cero
python3 fases/07-transformers-a-fondo/02-self-attention-desde-cero/code/main.py

# Lección 05 — transformer completo
python3 fases/07-transformers-a-fondo/05-transformer-completo/code/main.py

# Lección 14 — capstone (entrenamiento)
python3 fases/07-transformers-a-fondo/14-construye-un-transformer-capstone/code/main.py
```

Si los tres demos terminan con código 0, la fase está aprobada.

## Por qué importan los transformers

| Tarea | Antes de 2017 | Después de 2017 |
|---|---|---|
| Traducción | LSTM seq2seq + atención | Transformer |
| Clasificación de texto | CNN/LSTM + embeddings | BERT |
| Generación de texto | n-gramas, LSTM | GPT |
| Visión | CNN | ViT, híbrido CNN-Transformer |
| Audio | HMM + GMM | Whisper |
| Multimodal | Embeddings concatenados | CLIP, LLaVA |

## Conexión con otras fases

- **Entrada** → [Fase 5 — NLP](../05-nlp-fundamentos-a-avanzado/README.md)
  y [Fase 3 — Núcleo de Deep Learning](../03-nucleo-deep-learning/README.md).
- **Salida natural** → [Fase 8 — IA generativa](../08-ia-generativa/README.md)
  y [Fase 10 — LLMs desde cero](../10-llms-desde-cero/README.md).
- **Reuso en** → Fase 4 (ViT), Fase 6 (Whisper), Fase 9 (RLHF
  usa GPT), Fase 12 (multimodal con transformer).

## Recursos recomendados

- *Attention Is All You Need* — Vaswani et al., 2017 (paper canónico).
- *The Illustrated Transformer* — Jay Alammar.
- *The Annotated Transformer* — Harvard NLP.
- *Build a Large Language Model (From Scratch)* — Sebastian Raschka.
- *Hugging Face NLP Course* — sección de transformers.
- *Flash Attention paper* — Dao et al., 2022.
- *Scaling Laws for Neural Language Models* — Kaplan et al., 2020.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *attention*,
  *KV cache*, *RoPE*, *MoE*, *Flash Attention*.
- [Fase 10 — LLMs desde cero](../10-llms-desde-cero/README.md).
- [Fase 11 — Ingeniería de LLMs](../11-ingenieria-llms/README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
