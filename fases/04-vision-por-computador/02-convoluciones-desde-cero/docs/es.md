# Convoluciones desde cero

> El bloque constructor de toda CNN: una operacion simple (multiplicar y sumar) que detecta patrones locales. Apilada y combinada, aprende features de bordes a objetos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 01-fundamentos-de-imagen
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar `conv2d` con kernel, padding, stride.
- Implementar max pool y average pool.
- Aplicar kernels tipicos (Sobel, blur, sharpen).

## Constrúyelo

```python
def conv2d(x, kernel, padding=0, stride=1):
    H, W = x.shape
    kH, kW = kernel.shape
    if padding > 0:
        x = np.pad(x, padding)
    out_H = (H - kH) // stride + 1
    out_W = (W - kW) // stride + 1
    salida = np.zeros((out_H, out_W))
    for i in range(out_H):
        for j in range(out_W):
            salida[i, j] = (x[i:i+kH, j:j+kW] * kernel).sum()
    return salida
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-conv-design
fase: 04
leccion: 02
---

1. Clasificacion: Conv3x3+BN+ReLU+Pool stack.
2. Segmentacion: encoder-decoder (U-Net).
3. Mobile: depthwise separable.
4. Default: 3x3, padding='same'.
```

## Ejercicios

1. **Conv2d con multiples canales**: input (H,W,C_in),
   output (H,W,C_out).
2. **Dilated conv (atrous)**: insertar 'huecos' en el kernel.
3. **Desafio**: implementa una capa Conv2d + ReLU + MaxPool
   en el mini-framework de la L10.

## Lecturas recomendadas

- "Deep Learning" (Goodfellow et al.) cap. 9
- "A guide to convolution arithmetic for deep learning"
  (Dumoulin & Visin, 2016)
- CS231n: <http://cs231n.github.io/convolutional-networks/>

---

> 📚 **Adaptación al español** de la lección "[Convolutions from Scratch]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).