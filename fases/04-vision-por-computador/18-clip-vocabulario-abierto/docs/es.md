# CLIP y visión de vocabulario abierto

> La invencion que conecto vision y lenguaje: 400M pares (imagen, texto) de internet, contrastive learning, espacio compartido. Zero-shot clasifica cualquier cosa sin entrenar.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17-vision-auto-supervisada
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar text/image encoder mocks.
- Calcular similitud coseno.
- Hacer zero-shot classification.
- Diagnosticar cuando usar CLIP vs VLM.

## Constrúyelo

```python
def clip_similarity(img_emb, txt_emb):
    img = img_emb / np.linalg.norm(img_emb)
    txt = txt_emb / np.linalg.norm(txt_emb)
    return float(img @ txt)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-clip-elegir
fase: 04
leccion: 18
---

1. Zero-shot clasif: CLIP, SigLIP.
2. Retrieval: CLIP, DINOv2.
3. VQA / captioning: BLIP-2, LLaVA.
4. Open-vocab det: YOLO-World, Grounding DINO.
5. Visual chat: GPT-4V, Claude 3.5, LLaVA.
```

## Ejercicios

1. **Prompt ensembling**: promediar embeddings de varios
   templates.
2. **CLIP + zero-shot detection**: OWLv2 o Grounding DINO.
3. **Desafio**: hacer retrieval de imagenes custom con
   CLIP y evaluar con Recall@K.

## Lecturas recomendadas

- "CLIP" (Radford et al., 2021)
- "BLIP-2" (Li et al., 2023)
- "LLaVA" (Liu et al., 2023)
- open_clip: <https://github.com/mlfoundations/open_clip>

---

> 📚 **Adaptación al español** de la lección "[Open-Vocabulary CLIP]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).