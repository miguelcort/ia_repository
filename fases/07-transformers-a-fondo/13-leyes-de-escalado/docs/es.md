# 13 — Leyes de escalado

> Las scaling laws de Kaplan (2020) y Hoffmann (2022) definen cómo varía el loss con parámetros, datos, y cómputo. Guiaron el diseño de GPT-3, Chinchilla, y todos los LLMs modernos.

**Tipo:** Aprender
**Lenguajes:** Python
**Prerrequisitos:** 05-transformer-completo
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Entender las scaling laws de Kaplan et al. (2020) y
  Chinchilla / Hoffmann et al. (2022).
- Calcular el cómputo óptimo de entrenamiento para un
  budget dado.
- Diagnosticar cuándo un modelo está under-trained o
  over-sized.
- Aplicar las leyes a decisiones prácticas de training.

## El problema

Entrenar un LLM cuesta millones de dólares. ¿Cuántos
parámetros, cuántos tokens, y cuántos FLOPs? Las
scaling laws de Kaplan (2020) y Chinchilla (2022)
proporcionan fórmulas empíricas: el loss escala como
power law con parámetros, datos, y cómputo. La lección
cubre ambas y sus implicaciones prácticas.

## El concepto

**Kaplan et al. (2020).** Loss = E + (N_c / N)^α_N + (D_c / D)^α_D.
El loss tiene un piso E y decrece como power law con
parámetros N y datos D. Conclusión: "más grande siempre
es mejor" (con suficiente cómputo). Limitación: asume
que el cómputo es ilimitado.

**Chinchilla / Hoffmann et al. (2022).** Loss =
E + A / (N^α + D^β) + ... El cómputo total C = 6ND (forward
+ backward pass). Para C fijo, el óptimo es N ∝ C^0.5 y
D ∝ C^0.5. Conclusión: **debe asignarse igual compute a
parámetros y datos**. El modelo Gopher de 280B estaba
over-sized para sus 300B tokens; Chinchilla (70B, 1.4T
tokens) logró mejor loss con 4x menos compute.

**Implicaciones prácticas.**

- Para un budget de 1e24 FLOPs, entrenar un modelo
  Chinchilla-optimal: N ≈ 70B, D ≈ 1.4T tokens.
- Para 1e21 FLOPs, N ≈ 1B, D ≈ 20B tokens.
- Para fine-tuning (no pre-training), las scaling
  laws no aplican directamente.

**Otras scaling laws.**

- **Leyes de datos:** la calidad importa más que la
  cantidad. DataComp-LM muestra que filtrar el dataset
  mejora más que aumentar su tamaño.
- **Leyes de inference:** scaling en inference (más
  compute per query) sigue power laws hasta cierto
  punto. Distillation comprime sin perder calidad.
- **Leyes multimodales:** vision-language y audio-language
  también siguen power laws, con diferentes
  coeficientes.

**Cuándo las scaling laws aplican.**

- Pre-entrenamiento desde cero con datasets
  suficientemente grandes (> 1B tokens).
- Arquitecturas de transformer estándar.
- No aplican a fine-tuning, RLHF, o modelos
  multimodales donde hay componentes no-lineales
  adicionales.

**Emergent abilities.** A ciertos tamaños, los modelos
muestran habilidades nuevas (few-shot learning, chain-of-
thought reasoning). Las scaling laws no predicen cuándo
emergen. Wei et al. (2022) y Schaeffer et al. (2023)
discuten si son realmente emergentes o artefactos de
métricas.

**Trampas.**

- **Más grande siempre es mejor:** NO. Chinchilla
  muestra que necesitas datos proporcionales.
- **Ignorar calidad de datos:** 1T tokens de internet
  random < 100B tokens de datos filtrados. DataComp
  muestra esto claramente.
- **Aplicar scaling laws a fine-tuning:** no se
  aplican. Fine-tuning es un regime diferente.

## Constrúyelo

```python
import math


def kaplan_loss(N, D, E=1.5, A=2.5e3, alpha_N=0.34, alpha_D=0.28):
    """Loss según Kaplan et al. (2020)."""
    return E + A * (1 / N ** alpha_N + 1 / D ** alpha_D)


def chinchilla_optimal(C, k=1.2):
    """Para compute budget C, optimal N y D según Chinchilla.
    C = 6ND, óptimo: N = (C / 6 / k) ^ 0.5, D = k * N."""
    N_opt = (C / (6 * k)) ** 0.5
    D_opt = k * N_opt
    return N_opt, D_opt


def total_flops(N, D):
    """FLOPs totales para entrenar: 6ND (forward + backward)."""
    return 6 * N * D
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

Eres un asistente que ayuda con decisiones de escalado.
Recibirás el budget de compute y los recursos. Tu trabajo:

1. Para pre-entrenamiento desde cero: usar Chinchilla
   para N y D óptimos.
2. Compute budget C → N_opt ≈ (C/6)^0.5, D_opt ≈ 20 * N_opt
   (Chinchilla ratio).
3. Si datos son el cuello: 1T tokens es típico para
   LLMs SOTA.
4. Si compute es el cuello: pre-entrenar más pequeño
   con más datos.
5. Quality > quantity: filtrar datos mejora más que
   aumentar su tamaño.
6. Para fine-tuning: no aplicar scaling laws; usar
   recomendaciones específicas de la tarea.
7. Monitorear: si training loss no baja, modelo
   under-sized; si baja pero val loss no, over-sized.
```

## Ejercicios

1. **Chinchilla**: implementa el cálculo de N y D
   óptimos para tu budget.
2. **Loss curve**: visualiza cómo varía el loss con
   N, D, y C.
3. **Desafío**: diseña un plan de pre-entrenamiento
   para 1e24 FLOPs usando Chinchilla.

## Lecturas recomendadas

- *Scaling Laws for Neural Language Models* — Kaplan et
  al., 2020.
- *Training Compute-Optimal Large Language Models
  (Chinchilla)* — Hoffmann et al., 2022.
- *DataComp-LM* — Li et al., 2024.
- *Are Emergent Abilities of Large Language Models a
  Mirage?* — Schaeffer et al., 2023.

---

> 📚 **Adaptación al español** de la lección "[Scaling Laws]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
