# SAM3 y segmentación de vocabulario abierto

> Foundation model de segmentacion: SAM (imagen), SAM 2 (video), SAM 3 (open-vocab con texto). Zero-shot, interactivo, con prompts de puntos, boxes, mascaras. SA-1B: 1.1B mascaras.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 18-clip-vocabulario-abierto
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar ViT image encoder.
- Codificar prompts (puntos, texto).
- Generar mascaras zero-shot.
- Evaluar con IoU y mIoU.

## Constrúyelo

```python
def sam_prompt_encoder_point(points, labels):
    pos = np.array(points, dtype=np.float32) / 1024.0
    pos = (pos - 0.5) * 2
    lbl = np.array(labels, dtype=np.float32).reshape(-1, 1)
    feats = np.concatenate([pos, lbl], axis=-1)
    return feats @ W  # proyectar a 256 dim
```

## Úsalo

```bash
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

1. Zero-shot: SAM con prompt points/box.
2. Video: SAM 2.
3. Texto libre: SAM 3 o Grounded-SAM.
4. Mobile: MobileSAM.
5. Medical: Cellpose, Stardist.
```

## Ejercicios

1. **Multi-prompt**: combinar puntos foreground y
   background, refinar mascara iterativamente.
2. **Grounded-SAM**: integrar Grounding DINO con SAM.
3. **Desafio**: usar SAM para segmentar tumores en un
   dataset de medical images con zero-shot.

## Lecturas recomendadas

- "Segment Anything" (Kirillov et al., 2023)
- "SAM 2" (Ravi et al., 2024)
- Grounded-SAM: <https://github.com/IDEA-Research/Grounded-Segment-Anything>

---

> 📚 **Adaptación al español** de la lección "[SAM3 and Open-Vocabulary Segmentation]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).