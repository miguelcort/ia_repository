# Procesos estocasticos

> Casi todo en ML es un proceso estocastico: caminatas, MCMC, diffusion, RL.

**Tipo:** Aprender
**Lenguajes:** Python
**Prerrequisitos:** 06-probabilidad-y-distribuciones
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Simular caminatas aleatorias.
- Construir cadenas de Markov.
- Diagnosticar cuando aplicar MCMC.

## Constrúyelo

```python
import numpy as np


def caminata_aleatoria(n_pasos, semilla=0):
    rng = np.random.default_rng(semilla)
    return np.cumsum(rng.normal(0, 1, n_pasos))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-stoch-elegir
fase: 01
leccion: 22
---

1. Eventos raros: Poisson.
2. Movimientos continuos: Brownian.
3. Memoria 1: Markov.
4. Muestreo complejo: MCMC.
5. Series largas: ARIMA.
```

## Ejercicios

1. **Poisson**: simula llegadas a una cola y verifica la distribucion
   de tiempos entre llegadas.
2. **MCMC**: implementa Metropolis-Hastings para una Normal 2D.
3. **Desafio**: implementa un ARIMA(1, 1, 1) desde cero.

## Lecturas recomendadas

- "Stochastic Processes" (Ross)
- PyMC: <https://www.pymc.io/>

---

> 📚 **Adaptación al español** de la lección "[Stochastic Processes]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).