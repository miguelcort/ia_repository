# Inicialización de pesos

> El primer paso del entrenamiento define la dinamica. Una buena inicializacion (He, Xavier, Lecun) mantiene la varianza estable y permite entrenar redes profundas.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-redes-multicapa
**Tiempo estimado:** ~25 minutos

## Objetivos de aprendizaje

- Implementar inicializaciones Zero, Random, Xavier, He, Lecun.
- Comparar estabilidad de varianza de activaciones.
- Diagnosticar vanishing/exploding por mala inicializacion.

## Constrúyelo

```python
def he(n_in, n_out, semilla=0):
    rng = np.random.default_rng(semilla)
    escala = np.sqrt(2.0 / n_in)
    return rng.normal(scale=escala, size=(n_in, n_out))
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-init
fase: 03
leccion: 08
---

1. ReLU/GELU: He (kaiming).
2. Sigmoid/tanh: Xavier (glorot).
3. SELU: Lecun.
4. Biases: 0 (default), 0.01 si ReLU con riesgo dead.
5. NUNCA pesos a cero.
```

## Ejercicios

1. **Truncated normal**: limita los pesos a |w| < 2*sigma.
2. **Orthogonal init**: para RNN, W = Q de QR.
3. **Desafio**: visualiza distribuciones de activaciones capa
   por capa para 3 inicializaciones.

## Lecturas recomendadas

- "Understanding the difficulty of training deep feedforward
  neural networks" (Glorot & Bengio, 2010)
- "Delving Deep into Rectifiers" (He et al., 2015)
- "Efficient BackProp" (LeCun et al., 1998)

---

> 📚 **Adaptación al español** de la lección "[Weight Initialization]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).