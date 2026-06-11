# Leyes de escalado

> Chinchilla scaling laws: L(N, D) = a·N^(-α) + b·D^(-β) + e. Compute C ≈ 6ND. Ratio óptimo N:D ~ 1.4:1 (20:1 tokens:params). Chinchilla 70B / 1.4T > Gopher 280B / 380B. Overtraining: Phi-3 3.8B en 3.3T tokens. Llama 3.1 405B en 15.6T. Limitaciones: no extrapolan a multimodal, post-training, inferencia.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 07/05-transformer-completo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar Chinchilla loss formula.
- Calcular N_opt y D_opt dado compute budget.
- Comparar Kaplan vs Chinchilla.
- Diagnosticar overtraining vs compute-optimal.

## Constrúyelo

```python
def chinchilla_loss(N, D, a=406.4, b=410.7, alpha=0.34, beta=0.28, e=1.69):
    return a * (N ** -alpha) + b * (D ** -beta) + e
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-scaling-laws
fase: 07
leccion: 13
---

1. Chinchilla: L = a*N^-alpha + b*D^-beta + e.
2. N_opt = (alpha/beta) * D_opt.
3. Phi overtrained, Llama compute-optimal.
4. Frontier: 100-400B params, 1-15T tokens.
5. Limitaciones: multimodal, post-training.
```

## Ejercicios

1. **Compute allocation**: dado budget de 1e24
   FLOPs, calcular N_opt y D_opt.
2. **Overtraining**: entrenar Phi-mini en
   100B tokens y medir perplejidad.
3. **Desafio**: predecir costo de pre-entrenar
   un modelo 1T params con scaling laws.

## Lecturas recomendadas

- "Scaling Laws for Neural Language Models" (Kaplan et al., 2020)
- "Training Compute-Optimal Large Language Models" (Hoffmann et al., 2022)
- "Scaling Laws for Autoregressive Generative Modeling" (Henighan et al., 2020)

---

> 📚 **Adaptación al español** de la lección "[Scaling Laws]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).