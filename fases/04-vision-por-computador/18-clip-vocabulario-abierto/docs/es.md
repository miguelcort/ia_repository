# 18 — CLIP: open-vocabulary visual understanding

> CLIP (Contrastive Language-Image Pre-training) entrena un modelo para asociar imágenes con descripciones de texto. Permite zero-shot classification en cualquier categoría.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17-vision-auto-supervisada
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar CLIP desde cero (image + text encoders con
  contrastive loss).
- Aplicar zero-shot classification con CLIP preentrenado.
- Construir sistemas de búsqueda y recuperación
  cross-modal.
- Diagnosticar limitaciones de CLIP.

## El problema

Los modelos de clasificación tradicionales solo conocen
las clases con las que fueron entrenados. CLIP (Radford
et al., 2021) aprende a asociar imágenes y descripciones
de texto en un espacio conjunto. Una vez entrenado, puede
clasificar imágenes en **cualquier categoría** simplemente
dando los nombres como texto — sin reentrenamiento. Es la
base de DALL-E, Stable Diffusion (como condicionamiento),
y muchos sistemas de búsqueda y recomendación visual.

## El concepto

**Arquitectura.**

- **Image encoder:** ViT (o ResNet) que produce un
  embedding de la imagen.
- **Text encoder:** transformer que produce un embedding
  del texto.
- **Proyección:** ambos embeddings se proyectan al mismo
  espacio d-dimensional.
- **Cosine similarity:** la similitud se mide con coseno.

**Contrastive pre-training.** Para un batch de N pares
(imagen, texto), se predice cuál de los N textos
corresponde a cada imagen. La matriz NxN de similitudes
es la entrada de una cross-entropy simétrica (image-to-
text y text-to-image).

**Zero-shot classification.** Dada una imagen y K
categorías candidatas como texto ("a photo of a {cat}"),
se calculan las K similitudes imagen-texto y se elige la
más alta. Sin reentrenamiento.

**Aplicaciones.**

- **Zero-shot classification:** para clases nuevas.
- **Búsqueda cross-modal:** texto → imágenes, imagen →
  texto.
- **Filtrado de datasets:** descartar imágenes que no
  matchean un prompt.
- **Robust image classifier:** CLIP es muy robusto a
  distribuciones nuevas.
- **Image generation conditioning:** Stable Diffusion
  usa CLIP text encoder.

**Modelos CLIP.**

- **OpenAI CLIP:** ViT-L/14, ViT-H/14. Open weights.
- **OpenCLIP:** re-implementación open source, modelos
  más grandes (ViT-bigG/14).
- **SigLIP:** variante con sigmoid loss, mejor para
  training.
- **DFN (Data Filtering Networks):** CLIP entrenado con
  data filtering, mejor calidad.

**Trampas.**

- **CLIP no es perfecto en tareas finas:** clasificación
  específica (razas de perros, modelos de autos) requiere
  fine-tuning.
- **Sesgo de los datos:** CLIP aprende los sesgos de su
  dataset de training (LAION).
- **Distribución diferente:** zero-shot falla si las
  imágenes de producción son muy distintas de las de
  training.

## Constrúyelo

```python
import numpy as np


def clip_image_embed(image, image_encoder, projection):
    """Proyecta una imagen al espacio CLIP."""
    features = image_encoder(image)
    emb = features @ projection
    return emb / np.linalg.norm(emb, axis=-1, keepdims=True)


def clip_text_embed(text, text_encoder, projection):
    """Proyecta texto al espacio CLIP."""
    features = text_encoder(text)
    emb = features @ projection
    return emb / np.linalg.norm(emb, axis=-1, keepdims=True)


def clip_zero_shot(image_emb, text_embs, class_names):
    """Clasifica la imagen con la clase cuyo texto tiene
    mayor similitud coseno."""
    similarities = image_emb @ text_embs.T
    return class_names[np.argmax(similarities)]


def clip_contrastive_loss(image_embs, text_embs, tau=0.07):
    """Symmetric InfoNCE loss para CLIP."""
    n = image_embs.shape[0]
    logits = image_embs @ text_embs.T / tau
    labels = np.arange(n)
    # Image-to-text
    loss_i2t = -np.mean(np.log(
        np.exp(logits[np.arange(n), labels]) /
        np.exp(logits).sum(axis=1)
    ))
    # Text-to-image
    loss_t2i = -np.mean(np.log(
        np.exp(logits[labels, np.arange(n)]) /
        np.exp(logits).sum(axis=0)
    ))
    return (loss_i2t + loss_t2i) / 2
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-clip
fase: 04
leccion: 18
---

Eres un asistente que ayuda a aplicar CLIP. Recibirás la
tarea y los datos disponibles. Tu trabajo:

1. Si quieres zero-shot classification: usar
   `open_clip` con ViT-bigG/14.
2. Si quieres búsqueda cross-modal: CLIP para embed,
   FAISS para búsqueda.
3. Si quieres filtrar imágenes: usar CLIP para
   descartar imágenes que no matchean el prompt.
4. Si necesitas tareas específicas (razas de perros):
   fine-tuning con linear probe.
5. Para producción: usar SigLIP o DFN (mejor
   calidad y robustez).
6. Advertir contra CLIP para tareas muy específicas
   (mejor fine-tuning).
7. Advertir contra sesgo de datos: CLIP aprende los
   sesgos de su training set.
```

## Ejercicios

1. **Zero-shot**: implementa zero-shot classification con
   CLIP preentrenado en un dataset personalizado.
2. **Búsqueda**: indexa 10k imágenes con CLIP y haz
   búsqueda por texto.
3. **Desafío**: fine-tunea CLIP en un dataset de
   clasificación específica y compara con zero-shot.

## Lecturas recomendadas

- *Learning Transferable Visual Models From Natural
  Language Supervision (CLIP)* — Radford et al., 2021.
- *SigLIP: Sigmoid Loss for Language Image Pre-Training* —
  Zhai et al., 2023.
- *DFN: Data Filtering Networks* — Fang et al., 2023.
- OpenCLIP: <https://github.com/mlfoundations/open_clip>.

---

> 📚 **Adaptación al español** de la lección "[CLIP: Open-Vocabulary Visual Understanding]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
