# Fase 10 — LLMs desde cero

> Construir, entrenar y entender modelos de lenguaje grandes.

La fase anterior (Fase 7) cubrió los *transformers* como arquitectura
general. Esta fase **entrega el libro completo** sobre LLMs: cómo se
tokeniza texto, cómo se entrena un GPT desde cero, cómo se hace
*fine-tuning* con instrucciones, cómo se alinea con preferencias
humanas (RLHF, DPO), cómo se cuantiza, cómo se sirve eficientemente,
y cómo se leen los papers de arquitecturas modernas (DeepSeek-V3,
Jamba, EAGLE-3, NSA, MTP).

La fase está organizada en **cuatro bloques**. El **bloque 1**
(lecciones 1–4) cubre los **componentes fundamentales**: tokenizers,
pipelines de datos y pre-entrenamiento de un mini-GPT (124M). El
**bloque 2** (5–10) entra al **alineamiento y evaluación**:
entrenamiento distribuido (FSDP, DeepSpeed), SFT, RLHF, DPO,
Constitutional AI y benchmarks de evaluación. El **bloque 3** (11–
13) trata la **producción**: cuantización, optimización de
inferencia y un pipeline completo. El **bloque 4** (14–23) es
**arquitecturas modernas**: walkthroughs de modelos abiertos,
speculative decoding, attention variants (differential, sparse,
native), multi-token prediction, DualPipe, Jamba, gradient
checkpointing.

## Índice de lecciones

### Bloque 1 — Componentes fundamentales

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [Tokenizers: BPE, WordPiece, SentencePiece](01-tokenizers/) | Construir | Algoritmos y tradeoffs de tokenización. |
| 02 | [Construir un tokenizer desde cero](02-construyendo-un-tokenizer/) | Construir | BPE paso a paso con tests. |
| 03 | [Pipelines de datos para pre-entrenamiento](03-pipelines-de-datos/) | Construir | Deduplicación, filtrado, *packing* y *shuffling*. |
| 04 | [Pre-entrenar un mini-GPT (124M)](04-pre-training-mini-gpt/) | Construir | GPT-2 pequeño en TinyStories o FineWeb. |

### Bloque 2 — Alineamiento y evaluación

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 05 | [Entrenamiento distribuido: FSDP, DeepSpeed](05-scaling-y-distribuido/) | Construir | Data, tensor y pipeline parallelism. |
| 06 | [Instruction tuning — SFT](06-instruction-tuning-sft/) | Construir | Formato Alpaca/ChatML, *masked loss*. |
| 07 | [RLHF — reward model + PPO](07-rlhf/) | Construir | Pipeline completo con *reward model* y PPO. |
| 08 | [DPO — Direct Preference Optimization](08-dpo/) | Construir | DPO desde cero, sin RL. |
| 09 | [Constitutional AI y self-improvement](09-constitutional-ai-y-mejora-self/) | Construir | RLAIF, *critique-revise* loop. |
| 10 | [Evaluación — benchmarks y evals](10-evaluacion/) | Construir | MMLU, GSM8K, AlpacaEval, MT-Bench. |

### Bloque 3 — Producción

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 11 | [Cuantización: INT8, GPTQ, AWQ, GGUF](11-cuantizacion/) | Construir | PTQ vs QAT, *weight-only* y *k-quant*. |
| 12 | [Optimización de inferencia](12-optimizacion-de-inferencia/) | Construir | KV cache, batch, paginación y vLLM. |
| 13 | [Pipeline completo de LLM (capstone)](13-pipeline-completo-de-llm/) | Construir | Tokenizer → train → SFT → serve. |

### Bloque 4 — Arquitecturas modernas

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 14 | [Open models: walkthroughs de arquitectura](14-walkthroughs-de-arquitectura-de-modelos-open/) | Aprender | LLaMA, Mistral, Qwen, Gemma. |
| 15 | [Speculative decoding y EAGLE-3](15-speculative-decoding-eagle3/) | Construir | Draft-verify, multi-token. |
| 16 | [Differential attention (V2)](16-atencion-diferencial-v2/) | Construir | Cancelación de ruido, *diff attn*. |
| 17 | [Native sparse attention (DeepSeek NSA)](17-atencion-nativa-dispersa/) | Construir | Compresión + selección dispersa. |
| 18 | [Multi-token prediction (MTP)](18-multi-token-prediction/) | Construir | Predicción de múltiples tokens. |
| 19 | [DualPipe parallelism](19-dualpipe-y-paralelismo/) | Aprender | Solapamiento de *forward/backward*. |
| 20 | [DeepSeek-V3: walkthrough de arquitectura](20-walkthrough-de-deepseek-v3/) | Aprender | MoE, MLA, MTP y FP8. |
| 21 | [Jamba — SSM-Transformer híbrido](21-jamba-hibrido-ssm-transformer/) | Aprender | Mamba + transformer. |
| 22 | [Inferencia asíncrona y Hogwild!](22-async-hogwild-inference/) | Construir | Batch asíncrono, throughput. |
| 23 | [Gradient checkpointing y recompute](23-gradient-checkpointing-y-recompute/) | Construir | Memoria vs cómputo. |

## Prerrequisitos

- **Fases 5, 7 y 9** completas.
- Conocimiento sólido de PyTorch.
- GPU con 16+ GB VRAM (24+ GB recomendado para SFT/DPO).
- Opcional: experiencia con `accelerate` y `deepspeed`.

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Construir** un tokenizer BPE desde cero y compararlo con
  SentencePiece y tiktoken.
- **Pre-entrenar** un mini-GPT de 124M parámetros en
  TinyStories/FineWeb.
- **Hacer SFT** sobre un modelo pre-entrenado con datos Alpaca o
  ChatML.
- **Aplicar DPO y RLHF** y entender las diferencias prácticas.
- **Evaluar** LLMs con benchmarks estándar y *llm-as-judge*.
- **Cuantizar** un modelo a INT8/INT4 con GPTQ, AWQ o GGUF.
- **Servir** un LLM con vLLM, TGI o llama.cpp.
- **Leer** los papers de arquitecturas modernas (DeepSeek-V3,
  Jamba, EAGLE-3) y entender sus contribuciones.

## Stack y herramientas

- **PyTorch** y **transformers** (Hugging Face).
- **datasets** y **accelerate** para pipelines de datos y
  entrenamiento.
- **tokenizers** y **sentencepiece** para tokenización.
- **trl** (Transformer Reinforcement Learning) para SFT/DPO/RLHF.
- **deepspeed** y **fsdp** para entrenamiento distribuido.
- **vllm**, **TGI**, **llama.cpp**, **ollama** para servir.
- **bitsandbytes** y **auto-gptq** para cuantización.
- **wandb** para tracking.
- **lm-eval-harness** para evaluación.

## Conceptos clave

| Concepto | Aparece en | Reaparece en |
|---|---|---|
| **BPE** | Lecciones 01, 02 | Cualquier LLM. |
| **Pre-training** | Lección 04 | Base de todo. |
| **SFT** | Lección 06 | Fase 11 (fine-tuning en producción). |
| **RLHF / DPO** | Lecciones 07, 08 | Fase 9 + 10. |
| **FSDP** | Lección 05 | Entrenamiento de modelos grandes. |
| **Cuantización** | Lección 11 | Inferencia en hardware limitado. |
| **Speculative decoding** | Lecciones 12, 15 | Optimización de inferencia. |
| **MoE / MLA / MTP** | Lección 20 | Arquitecturas modernas. |

## Cómo estudiar esta fase

1. **Empieza por el tokenizer (lecciones 1–2).** Es la base de
   todo lo demás.
2. **El capstone (lección 13) integra toda la fase.** Hazlo al
   final, no en paralelo.
3. **Para entrenamiento distribuido, lee la documentación oficial
   primero.** `accelerate` y `deepspeed` cambian rápidamente.
4. **DPO es más simple que RLHF y a menudo mejor.** Úsalo como
   primera opción a menos que tengas razones específicas para
   RLHF.
5. **La cuantización es práctica, no teórica.** Prueba INT4 con
   tu modelo y mide la pérdida de calidad vs. el ahorro de
   memoria.

## Verificación de progreso

```bash
# Lección 04 — mini-GPT pre-entrenado
python3 fases/10-llms-desde-cero/04-pre-training-mini-gpt/code/main.py

# Lección 08 — DPO en un modelo Alpaca
python3 fases/10-llms-desde-cero/08-dpo/code/main.py

# Lección 13 — pipeline completo
python3 fases/10-llms-desde-cero/13-pipeline-completo-de-llm/code/main.py
```

Si los tres demos terminan con código 0, la fase está aprobada.

## Cuándo usar cada técnica de alineamiento

| Objetivo | Técnica | Lección |
|---|---|---|
| Seguir instrucciones generales | SFT | 06 |
| Alinear con preferencias humanas | DPO (preferida) | 08 |
| Maximizar una señal compleja (humana) | RLHF | 07 |
| Mejorar con feedback de IA | Constitutional AI / RLAIF | 09 |
| Evaluar comportamiento | Benchmarks + LLM-judge | 10 |

## Conexión con otras fases

- **Entrada** → [Fase 7 — Transformers](../07-transformers-a-fondo/README.md),
  [Fase 9 — RL](../09-aprendizaje-por-refuerzo/README.md).
- **Salida natural** → [Fase 11 — Ingeniería de LLMs](../11-ingenieria-llms/README.md)
  (fine-tuning, RAG, agentes) y [Fase 14 — Ingeniería de
  agentes](../14-ingenieria-agentes/README.md).
- **Reuso en** → Fase 12 (multimodal), Fase 17 (producción), Fase
  19 (capstone).

## Recursos recomendados

- *Build a Large Language Model (From Scratch)* — Sebastian Raschka.
- *Hands-On Large Language Models* — Jay Alammar.
- *The Illustrated GPT-2* — Jay Alammar.
- *Hugging Face NLP Course* — capítulos de LLM.
- *Direct Preference Optimization paper* — Rafailov et al., 2023.
- *DeepSeek-V3 Technical Report* — DeepSeek, 2024.
- *Jamba paper* — AI21, 2024.

## Véase también

- [glosario/terminos.md](../../glosario/terminos.md) — *BPE*,
  *RLHF*, *DPO*, *SFT*, *FSDP*, *KV cache*.
- [Fase 11 — Ingeniería de LLMs](../11-ingenieria-llms/README.md).
- [Fase 14 — Ingeniería de agentes](../14-ingenieria-agentes/README.md).
- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.

---

> 📚 **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
