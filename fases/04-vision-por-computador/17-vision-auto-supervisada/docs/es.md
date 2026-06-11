# Visión auto-supervisada

> Aprender features sin etiquetas: pretexto tasks (contrastive, masked, clustering) producen representaciones comparables a ImageNet preentrenado. DINOv2 y MAE marcan el estado del arte.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14-vision-transformers
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar augmentaciones dobles.
- Implementar SimCLR contrastive loss.
- Implementar MAE mask.
- Diagnosticar cuando usar SSL.

## Constrúyelo

```python
def simclr_loss(z1, z2, temperatura=0.1):
    z = np.vstack([z1, z2])
    sim = z @ z.T / temperatura
    exp = np.exp(sim - sim.max(axis=-1, keepdims=True))
    probs = exp / exp.sum(axis=-1, keepdims=True)
    loss = 0
    for i in range(len(z1)):
        loss -= np.log(probs[i, i + len(z1)] + 1e-10)
        loss -= np.log(probs[i + len(z1), i] + 1e-10)
    return loss / (2 * len(z1))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-ssl
fase: 04
leccion: 17
---

1. Muchos datos: DINOv2, MAE.
2. Pocos labels: linear probe DINOv2.
3. Dominio custom: MAE/DINO desde cero.
4. Text-image: CLIP, SigLIP.
5. Segmentar: SAM.
```

## Ejercicios

1. **MoCo**: implementar cola de negativos con
   momentum encoder.
2. **BYOL**: self-distillation sin negativos.
3. **Desafio**: preentrenar MAE en un dataset custom
   (e.g. tus fotos) y linear-probe en CIFAR-10.

## Lecturas recomendadas

- "SimCLR" (Chen et al., 2020)
- "MAE" (He et al., 2022)
- "DINOv2" (Oquab et al., 2023)
- lightly: <https://docs.lightly.ai/>

---

> 📚 **Adaptación al español** de la lección "[Self-Supervised Vision]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).