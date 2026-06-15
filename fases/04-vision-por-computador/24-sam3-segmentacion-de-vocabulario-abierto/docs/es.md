# 24 — SAM 3 y segmentación open-vocabulary

> SAM 3 (Segment Anything Model 3) lleva la segmentación a un nivel más allá: promptable con texto, puntos, y bounding boxes. Detecta cualquier objeto nombrado en lenguaje natural.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07-segmentacion-semantica-unet,
                  18-clip-vocabulario-abierto
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Usar SAM 3 con prompts de texto, puntos, y bounding
  boxes.
- Combinar SAM con detección de objetos para segmentar
  instancias.
- Diagnosticar trade-offs entre velocidad y calidad.
- Conocer variantes: SAM 2 (video), SAM-Med-3D (médico).

## El problema

SAM (Kirillov et al., 2023) revolucionó la segmentación
al permitir segmentar **cualquier objeto** con solo un
prompt (puntos, box, máscara). SAM 2 (Ravi et al., 2024)
extiende a video. SAM 3 añade prompts de **texto**
estilo open-vocabulary. La lección cubre el uso práctico
de SAM y sus variantes.

## El concepto

**SAM 1 (Segment Anything Model).** Modelo de
segmentación promptable. Acepta varios tipos de prompt:

- **Puntos (positivos/negativos):** "este punto está en
  el objeto" / "este punto no está".
- **Bounding boxes:** "este objeto está aquí".
- **Máscaras:** refinar una máscara existente.

Devuelve máscaras binarias con score de confianza.

**Arquitectura SAM.**

- **Image encoder:** ViT-H preentrenado con MAE.
  Procesa la imagen una vez.
- **Prompt encoder:** embeddings de los prompts.
- **Mask decoder:** transformer que combina
  embeddings de la imagen y los prompts, produce
  máscaras.
- **Mask head:** MLP pequeño que produce la máscara
  final.

**Uso típico.**

```python
from segment_anything import sam_model_registry, SamPredictor

sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h.pth")
predictor = SamPredictor(sam)
predictor.set_image(image)
masks, scores, _ = predictor.predict(
    point_coords=np.array([[x, y]]),
    point_labels=np.array([1]),
)
```

**SAM 2 (video).** Añade un módulo de memoria para
tracking de la máscara a través de frames. Procesa
cada frame y propaga la máscara según la memoria.

**SAM 3 (open-vocabulary, 2024-2025).** Acepta **texto**
como prompt. Internamente usa un text encoder similar
a CLIP, integra con el image encoder, y produce
máscaras para las entidades nombradas en el texto.
Permite: `mask = sam3.predict(image, "the red car")`.

**Variantes especializadas.**

- **SAM-Med-3D:** para imágenes médicas 3D (CT, MRI).
- **SAM-Track:** tracking de objetos en video.
- **MobileSAM:** SAM ligero para edge.
- **EfficientSAM:** eficiencia mejorada con
  encoder más pequeño.

**Trampas.**

- **Imagen muy grande (> 1024x1024):** SAM tarda
  bastante. Redimensionar o usar SAM long.
- **Múltiples instancias:** SAM puede devolver
  múltiples máscaras. Elegir la de mayor score.
- **Objetos pequeños:** SAM funciona mejor con
  imágenes donde el objeto ocupa al menos 5% del
  frame.

## Constrúyelo

```python
import numpy as np


def sam3_zero_shot_mask(image, text_prompt, image_encoder,
                       text_encoder, mask_decoder,
                       threshold=0.5):
    """Zero-shot segmentation con SAM3.
    Devuelve la máscara binaria del objeto nombrado."""
    img_emb = image_encoder(image)
    txt_emb = text_encoder(text_prompt)
    mask_logits = mask_decoder(img_emb, txt_emb)
    mask = mask_logits > threshold
    return mask


def mask_to_bbox(mask):
    """Bounding box de una máscara binaria."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin, rmin, cmax + 1, rmax + 1]


def mask_iou(mask1, mask2):
    inter = np.sum(mask1 & mask2)
    union = np.sum(mask1 | mask2)
    return inter / max(union, 1e-12)
```

## Úsalo

```bash
pip install segment-anything
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-sam
fase: 04
leccion: 24
---

Eres un asistente que ayuda a usar SAM 3. Recibirás la
imagen, el prompt (texto, puntos, box), y la aplicación.
Tu trabajo:

1. Si el prompt es texto: usar SAM 3 con text encoder.
2. Si el prompt es puntos o box: usar SAM clásico.
3. Para video: SAM 2 con memoria.
4. Para medical: SAM-Med-3D.
5. Para edge: MobileSAM o EfficientSAM.
6. Preprocesar: resize a 1024x1024 max, normalizar
   con ImageNet stats.
7. Postprocesar: NMS sobre múltiples máscaras,
   threshold 0.5.
8. Visualizar: superponer la máscara en RGBA sobre la
   imagen original.
```

## Ejercicios

1. **SAM con texto**: implementa el wrapper para SAM 3
   con prompt de texto.
2. **Mask to bbox**: convierte una máscara SAM a
   bounding box.
3. **Desafío**: integra SAM con un detector de objetos
   para segmentar automáticamente todos los objetos
   detectados.

## Lecturas recomendadas

- *Segment Anything* — Kirillov et al., 2023.
- *SAM 2: Segment Anything in Images and Videos* — Ravi
  et al., 2024.
- *SAM 3: Segment Anything with Concepts* — Meta, 2024.
- segment-anything: <https://github.com/facebookresearch/segment-anything>.

---

> 📚 **Adaptación al español** de la lección "[SAM 3 and Open-Vocabulary Segmentation]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
