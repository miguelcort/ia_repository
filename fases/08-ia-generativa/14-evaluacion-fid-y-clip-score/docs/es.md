# Evaluación FID y CLIP score

> FID (Heusel 2017): Frechet distance entre Gaussianas de Inception pool3 features. ‖μ_r - μ_f‖² + Tr(Σ_r + Σ_f - 2·sqrt(Σ_r·Σ_f)). Más bajo = mejor. CLIP score: cosine similarity image-text, 0-1. IS: calidad + diversity, exp(KL). Métricas complementarias: precision/recall, LPIPS, FVD (video), FAD (audio). Human eval: gold standard. Limitaciones: bias a Inception/CLIP, no correlation perfecta con human eval.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 08/01-taxonomia-y-historia-de-modelos-generativos
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Frechet distance.
- Calcular FID, IS, CLIP score.
- Implementar precision/recall.
- Diagnosticar cuándo usar cada métrica.

## Constrúyelo

```python
def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean = sqrtm(sigma1 @ sigma2)
    return float(diff @ diff + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean.real))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-fid-clip
fase: 08
leccion: 14
---

1. FID: Frechet Inception Distance.
2. CLIP score: 0-1 cosine sim.
3. IS: calidad + diversity.
4. Precision/Recall: mode coverage.
5. Human eval gold standard.
```

## Ejercicios

1. **FID**: implementar Clean-FID y
   evaluar en custom model.
2. **CLIP score**: benchmark SD1.5 vs
   SDXL vs SD3.
3. **Desafio**: implementar VLM-as-judge
   con GPT-4V o LLaVA.

## Lecturas recomendadas

- "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium" (Heusel et al., 2017)
- "Learning Transferable Visual Models From Natural Language Supervision" (Radford et al., 2021)
- "Improved Precision and Recall Metric for Assessing Generative Models" (Kynkaanniemi et al., 2019)
- "DALL-E 2 vs DALL-E: A Comparison of Evaluation Metrics" (LAION)

---

> 📚 **Adaptación al español** de la lección "[Evaluation FID CLIP Score]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).