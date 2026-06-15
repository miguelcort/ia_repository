# 25 — Modelos visión-lenguaje (VLM)

> Los VLMs (LLaVA, Qwen-VL, InternVL) combinan un vision encoder con un LLM para responder preguntas sobre imágenes, describir, y razonar visualmente.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14-vision-transformers,
                  18-clip-vocabulario-abierto,
                  10-llms-desde-cero
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar la arquitectura ViT-MLP-LLM (estilo LLaVA).
- Hacer visual instruction tuning.
- Aplicar VLMs preentrenados (LLaVA, Qwen-VL) a tareas
  reales.
- Diagnosticar trade-offs entre resolución y costo.

## El problema

Un LLM solo procesa texto. Un VLM (Vision-Language Model)
añade visión: describe imágenes, responde preguntas sobre
ellas ("¿cuántas personas hay?"), razona visualmente
("¿qué está mal en este diagrama?"). LLaVA (Liu et al.,
2023) es la arquitectura canónica: ViT + MLP projector +
LLM. Qwen-VL, InternVL, GPT-4V, Gemini son VLMs SOTA en
2024-2025. La lección cubre la arquitectura y el flujo
de visual instruction tuning.

## El concepto

**Arquitectura ViT-MLP-LLM.**

- **Vision encoder:** ViT-L/14 (CLIP o SigLIP). Convierte
  imagen a tokens visuales.
- **MLP projector:** 2-3 capas lineales + GELU que proyectan
  los tokens visuales al espacio del LLM.
- **LLM:** Vicuna, LLaMA, Qwen, Mistral. Recibe los tokens
  visuales + texto y genera la respuesta.

**Visual instruction tuning.** Entrena el MLP projector
(con vision encoder y LLM congelados) sobre datos de
instrucciones visuales. Luego fine-tune el LLM (LoRA) y
el projector sobre datos de mayor calidad.

**LLaVA-1.5 (Liu et al., 2023).** ViT-L/14 + Vicuna-13B +
MLP projector. Preentrenado en CC3M-595K (image-text
pairs) y ajustado en LLaVA-Instruct-150K (visual
instructions). SOTA en benchmarks VQA en 2023.

**LLaVA-NeXT (2024).** Mejora la resolución de input a
cualquier tamaño (anyres). Performance SOTA en
benchmarks visuales.

**Qwen2-VL (2024).** ViT + Qwen2 LLM. Estado del arte
open-weight en VLM 2024-2025. Soporta video, multi-
imagen, y herramientas.

**InternVL3 (2024).** ViT + InternLM. SOTA en OCR y
documentos.

**Cuándo usar cada VLM.**

| Caso | Recomendación |
|---|---|
| Open-source SOTA 2024 | Qwen2-VL, InternVL3 |
| Réplica open de LLaVA | LLaVA-1.5 o LLaVA-NeXT |
| Producción hospedada | GPT-4o, Claude 3.5 Sonnet, Gemini 1.5 |
| Edge / mobile | MiniCPM-V, MobileVLM |
| OCR y documentos | Qwen-VL, InternVL3 |

**Tareas VLM.**

- **Image captioning:** describir la imagen en lenguaje
  natural.
- **VQA (Visual Question Answering):** responder
  preguntas sobre la imagen.
- **OCR:** leer texto en la imagen.
- **Document understanding:** entender formularios,
  tablas, gráficos.
- **Visual reasoning:** "qué pasaría si...", lógica visual.

**Trampas.**

- **Resolución baja:** VLMs pierden detalle. Usar
  modelos con any-resolution o multi-crop.
- **Alucinaciones:** los VLMs mienten sobre imágenes.
  Siempre verificar visualmente.
- **Costo:** llamar a GPT-4o en producción es caro.
  Considera self-host con Qwen2-VL.

## Constrúyelo

```python
import numpy as np


class VLM:
    """VLM minimal: ViT encoder + MLP projector + LLM."""

    def __init__(self, vision_encoder, llm, projector):
        self.vision_encoder = vision_encoder
        self.llm = llm
        self.projector = projector  # MLP ViT_dim -> LLM_dim

    def encode_image(self, image):
        """Convierte imagen a tokens visuales."""
        with np.errstate(divide="ignore"):
            features = self.vision_encoder(image)
        return features @ self.projector["W"] + self.projector["b"]

    def query(self, image, question):
        visual_tokens = self.encode_image(image)
        prompt = (
            f"<image>{visual_tokens}</image>"
            f"Pregunta: {question}\nRespuesta:"
        )
        return self.llm(prompt)

    def caption(self, image):
        return self.query(image, "Describe esta imagen en detalle.")
```

## Úsalo

```bash
pip install transformers torch
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-vlm
fase: 04
leccion: 25
---

Eres un asistente que ayuda a usar VLMs. Recibirás la tarea
y la aplicación. Tu trabajo:

1. Si necesitas SOTA open-source: Qwen2-VL o InternVL3.
2. Si necesitas réplica de LLaVA: LLaVA-1.5 o LLaVA-NeXT.
3. Si quieres hospedado: GPT-4o, Claude 3.5 Sonnet,
   Gemini 1.5.
4. Si edge: MiniCPM-V, MobileVLM.
5. Preprocesar: image a RGB, resize según modelo.
6. Prompt: claro, conciso, en español si el usuario
   escribe en español.
7. Verificar alucinaciones: revisar manualmente
   outputs críticos.
8. Limitar max_tokens para reducir costo.
```

## Ejercicios

1. **LLaVA desde cero**: implementa ViT + MLP + LLM
   con Hugging Face.
2. **VQA**: aplica un VLM preentrenado a un dataset de
   preguntas.
3. **Desafío**: fine-tunea LLaVA en un dataset de tu
   dominio.

## Lecturas recomendadas

- *Visual Instruction Tuning (LLaVA)* — Liu et al., 2023.
- *Qwen2-VL* — Wang et al., 2024.
- *InternVL3* — Chen et al., 2024.
- *GPT-4V(ision) System Card* — OpenAI, 2023.
- Hugging Face transformers: <https://huggingface.co/docs/transformers>.

---

> 📚 **Adaptación al español** de la lección "[Vision-Language Models (VLM)]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
