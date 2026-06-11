# Visual Autoregressive (VAR)

> VAR (NeurIPS 2024, ByteDance): visual autoregressive con predicción next-scale (coarse-to-fine). Multi-scale: 1×1 → 2×2 → 4×4 → ... → 256×256. 20× más rápido que AR tradicional, 10× que DiT. Calidad comparable a SD 1.5 en FID. Aplicaciones: image gen rápida, video (VideoVAR), 3D, hybrid con diffusion. Nuevo paradigma: next-scale vs next-token.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08/13-flow-matching-y-rectified-flows
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar patchify image a tokens.
- Implementar quantize con codebook.
- Implementar next-scale prediction loop.
- Comparar AR, VAR y diffusion.

## Constrúyelo

```python
def next_scale_prediction(generated_scales, n_scales, target_res):
    if not generated_scales:
        return np.array([[0]])
    last = generated_scales[-1]
    next_res = last.shape[0] * 2
    return np.zeros((next_res, next_res)) if next_res <= target_res else last
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-visual-autoregressive
fase: 08
leccion: 15
---

1. Next-scale: coarse-to-fine.
2. Multi-scale: 1x1 -> target.
3. 20x faster que AR.
4. Codebook tokens cuantizados.
5. VideoVAR, hybrid, 3D.
```

## Ejercicios

1. **VAR**: implementar y entrenar
   VAR pequeno en MNIST.
2. **Hybrid**: combinar VAR coarse con
   diffusion fine.
3. **Desafio**: extender VAR a video
   generation.

## Lecturas recomendadas

- "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction" (Chang et al., 2024)
- "Autoregressive Image Generation without Vector Quantization" (Li et al., 2024)
- "Scaling Autoregressive Models for Content-Rich Text-to-Image Generation" (Yu et al., 2022)

---

> 📚 **Adaptación al español** de la lección "[Visual Autoregressive VAR]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).