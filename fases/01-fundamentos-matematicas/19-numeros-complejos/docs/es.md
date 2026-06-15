# 19 — Números complejos para IA

> Los números complejos no son un capricho: FFT, procesamiento de señales y modelos de atención los necesitan.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 02-vectores-matrices-operaciones
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Realizar aritmética compleja en Python (operaciones,
  módulo, conjugado).
- Aplicar la fórmula de Euler para representar señales
  como suma de rotaciones en el plano complejo.
- Diagnosticar cuándo una tarea de IA se beneficia de
  aritmética compleja (FFT, atención, modelos de fase).

## El problema

La transformada de Fourier, base del procesamiento de audio y
de ciertos mecanismos de atención modernos, se define sobre
números complejos. Sin entender `i² = -1`, módulo, argumento y
conjugado, la FFT parece magia. La lección cubre el mínimo
indispensable para que el estudiante pueda leer código de FFT,
entender el *phasor* `e^{iωt}` y diagnosticar cuándo la
representación compleja es preferible a la real.

## El concepto

**Aritmética compleja.** Python tiene soporte nativo: `1+2j`,
`abs(1+2j) = sqrt(5)`, `(1+2j).conjugate() = 1-2j`. Las
operaciones siguen las reglas algebraicas usuales con `j² = -1`.
NumPy extiende esto a arrays: `np.array([1+2j, 3-4j])`.

**Geometría.** Un número complejo `z = a + bi` es un punto en
el plano. La **magnitud** `|z| = sqrt(a² + b²)` es la distancia
al origen. El **argumento** `arg(z) = atan2(b, a)` es el
ángulo. La **multiplicación** por `e^{iθ}` rota el punto
alrededor del origen en `θ` radianes.

**Fórmula de Euler.** `e^{iθ} = cos(θ) + i sin(θ)`. Es la
igualdad más importante de la matemática aplicada: conecta
exponenciales con trigonometría. Implicación: una señal
`cos(ωt)` se puede escribir como la parte real de
`e^{iωt}`. Una suma de señales se vuelve una suma de
exponenciales complejas.

**Conjugado y reflexión.** El conjugado de `z = a + bi` es
`z̄ = a - bi`. Refleja el punto a través del eje real.
`z · z̄ = |z|²` siempre es real y no negativo. En FFT, el
conjugado aparece al reconstruir la señal: la transformada
tiene simetría hermítica para señales reales.

**Por qué importa en IA.**

| Aplicación | Cómo usa complejos |
|---|---|
| **FFT** | Entrada/salida compleja; iFFT usa conjugado |
| **Procesamiento de audio** | Espectrograma es `|FFT|²` |
| **Convolución rápida** | Convolución = producto en Fourier |
| **Attention moderna** | Representaciones de fase (FNet, Hyena) |
| **Quantum ML** | Estados cuánticos son vectores complejos |

**FFT en una línea.** `np.fft.fft(x)` devuelve la transformada
como array complejo; `np.fft.ifft(X)` la inversa. Para una
señal real de longitud `N`, la FFT tiene `N//2 + 1` coeficientes
únicos (la otra mitad es el conjugado).

## Constrúyelo

```python
import numpy as np
import cmath
import math


def aritmetica_basica():
    """Demuestra operaciones básicas con complejos."""
    z1 = 1 + 2j
    z2 = 3 - 4j
    return {
        "suma": z1 + z2,
        "producto": z1 * z2,
        "magnitud_z1": abs(z1),
        "argumento_z1": cmath.phase(z1),
        "conjugado_z1": z1.conjugate(),
        "modulo_cuadrado": (z1 * z1.conjugate()).real,
    }


def euler_ejemplo(theta):
    """e^{i*theta} = cos(theta) + i sin(theta)"""
    z = cmath.exp(1j * theta)
    return {
        "complejo": z,
        "parte_real": z.real,
        "parte_imaginaria": z.imag,
        "magnitud": abs(z),
    }


def fft_demo(senal, fs=1.0):
    """Calcula la FFT de una señal y devuelve frecuencias y magnitudes."""
    X = np.fft.fft(senal)
    N = len(senal)
    freqs = np.fft.fftfreq(N, d=1.0 / fs)
    magnitudes = np.abs(X)
    return {
        "frecuencias": freqs[:N // 2],
        "magnitudes": magnitudes[:N // 2],
        "senal_reconstruida": np.fft.ifft(X).real,
    }
```

## Úsalo

```bash
cd code
python3 main.py
```

## Ejercicios

1. **Rotación**: implementa la rotación de un vector 2D
   multiplicándolo por `e^{iθ}` y verifica que preserva la
   magnitud.
2. **FFT**: aplica FFT a una señal de dos senos y verifica
   que los picos espectrales corresponden a las frecuencias
   reales.
3. **Convolución rápida**: implementa convolución por FFT
   (multiplicación punto a punto) y compara con convolución
   directa en `O(n²)`.

## Lecturas recomendadas

- *Complex Numbers from A to Z* — Andreescu & Andrica.
- *Fourier Analysis* — Stein & Shakarchi.
- NumPy FFT docs: <https://numpy.org/doc/stable/reference/routines.fft.html>.
- *The Scientist and Engineer's Guide to Digital Signal
  Processing* — Steven W. Smith, libro libre en línea.

---

> 📚 **Adaptación al español** de la lección "[Complex Numbers]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
