# Funciones de activación

> La no-linealidad que hace a las redes profundas. Elegir bien entre sigmoid, tanh, ReLU, GELU cambia la velocidad de convergencia y el tipo de patrones aprendibles.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-redes-multicapa
**Tiempo estimado:** ~25 minutos

## Objetivos de aprendizaje

- Implementar sigmoid, tanh, ReLU, leaky ReLU, GELU y softmax.
- Conocer rangos, derivadas y casos de uso.
- Diagnosticar dying ReLU y vanishing gradients.

## Constrúyelo

```python
def relu(z):
    return np.maximum(0, z)

def relu_derivada(z):
    return (z > 0).astype(float)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-activacion-elegir
fase: 03
leccion: 04
---

1. Hidden: ReLU (default), GELU (transformers).
2. Output binario: sigmoid + BCE.
3. Output multiclase: softmax + CE.
4. Compuertas RNN: sigmoid + tanh.
5. NUNCA sigmoid en hidden.
```

## Ejercicios

1. **PReLU**: implementa con alpha aprendido por capa.
2. **Swish/SiLU**: f(x) = x * sigmoid(x). Compara con ReLU.
3. **Desafio**: visualiza todas las activaciones y sus derivadas
   en el mismo plot.

## Lecturas recomendadas

- "Deep Learning" (Goodfellow et al.) cap. 6.3
- "Gaussian Error Linear Units" (Hendrycks & Gimpel, 2016)
- "Searching for Activation Functions" (Ramachandran et al., 2017)

---

> 📚 **Adaptación al español** de la lección "[Activation Functions]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).