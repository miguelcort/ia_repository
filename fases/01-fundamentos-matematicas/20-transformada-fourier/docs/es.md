# 20 — La transformada de Fourier

> Convierte señales en el dominio del tiempo al dominio de la frecuencia. Es la base de audio, convolución rápida y muchos modelos modernos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 19-numeros-complejos
**Tiempo estimado:** ~45 minutos

## Objetivos de aprendizaje

- Implementar la DFT (Discrete Fourier Transform) desde cero.
- Aplicar la FFT de NumPy y entender la diferencia con la
  DFT ingenua.
- Diagnosticar la simetría hermítica de la FFT para señales
  reales.
- Usar la convolución rápida por FFT y compararla con
  convolución directa.

## El problema

La convolución en el dominio del tiempo es `O(n²)`. La
misma operación en el dominio de la frecuencia es `O(n log n)`.
La FFT es la responsable de hacer eficiente la convolución, el
filtrado, el análisis espectral y cualquier tarea de
procesamiento de señales. Sin entender la FFT, las pipelines
de audio y video son cajas negras.

## El concepto

**DFT.** Para una señal `x` de longitud `N`, la DFT es:

```text
X[k] = Σ_{n=0}^{N-1} x[n] · e^{-2πi · k · n / N}
```

Devuelve `N` coeficientes complejos. La inversa reconstruye la
señal:

```text
x[n] = (1/N) · Σ_{k=0}^{N-1} X[k] · e^{2πi · k · n / N}
```

La DFT ingenua es `O(N²)`. La FFT (Cooley-Tukey) lo reduce a
`O(N log N)` dividiendo recursivamente en mitades pares e
impares.

**Simetría hermítica.** Si `x` es real, `X[N-k] = conj(X[k])`.
Por eso la FFT de una señal real tiene solo `N//2 + 1`
coeficientes únicos; la otra mitad es información redundante.
`np.fft.rfft` aprovecha esto.

**Convolución por FFT.** La convolución `x * y` se calcula
como `ifft(fft(x) · fft(y))`. El truco es rellenar con ceros
para evitar *circular wraparound*: `len_result = len(x) +
len(y) - 1`. Convolución directa es `O(n²)`; por FFT es
`O(n log n)`.

**STFT (Short-Time Fourier Transform).** Para señales no
estacionarias (audio, vibraciones), divides la señal en
ventanas cortas (típicamente 20-50 ms), aplicas FFT a cada
ventana y obtienes un espectrograma `|STFT|²` que muestra
cómo varía el espectro con el tiempo. Es la base de
Whisper, librosa, y todos los modelos de audio modernos.

**Aplicaciones en IA moderna.**

| Aplicación | Cómo usa FFT |
|---|---|
| **Whisper** | Log-mel spectrograma = `log(FFT(ventana))` filtrado con banco Mel |
| **Visión por convolución** | Convolución por FFT para *kernels* grandes |
| **Speech enhancement** | Spectral gating en el dominio de la frecuencia |
| **Modelos de fase** | FNet, Hyena — reemplazan atención con FFT |

**Fftshift y frecuencias.** `np.fft.fftfreq(N, d=1/fs)` devuelve
frecuencias en el orden `[0, 1/N, 2/N, ..., 1/2, -1/2, ...]`.
`np.fft.fftshift` las reordena a `[-1/2, ..., 0, ..., 1/2]`,
que es el formato que esperas ver en un espectrograma.

## Constrúyelo

```python
import numpy as np


def dft(x):
    """DFT ingenua, O(N^2)."""
    x = np.asarray(x, dtype=complex)
    N = len(x)
    n = np.arange(N)
    k = n.reshape(-1, 1)
    M = np.exp(-2j * np.pi * k * n / N)
    return (M @ x).flatten() if False else (M @ x.reshape(-1, 1)).flatten()


def fft(x):
    """Usa la implementación rápida de NumPy. O(N log N)."""
    return np.fft.fft(np.asarray(x, dtype=complex))


def convolve_fft(x, y):
    """Convolución por FFT con zero-padding para evitar wrap."""
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x) + len(y) - 1
    X = np.fft.fft(x, n)
    Y = np.fft.fft(y, n)
    return np.real(np.fft.ifft(X * Y))


def stft(senal, ventana=400, paso=160, fs=16000):
    """STFT con ventana Hann. Devuelve espectrograma de magnitud."""
    from numpy.fft import rfft
    w = np.hanning(ventana)
    n_frames = (len(senal) - ventana) // paso + 1
    spec = np.empty((n_frames, ventana // 2 + 1))
    for i in range(n_frames):
        frame = senal[i * paso : i * paso + ventana] * w
        spec[i] = np.abs(rfft(frame))
    return spec
```

## Úsalo

```bash
cd code
python3 main.py
```

## Ejercicios

1. **DFT desde cero**: implementa DFT y compara con
   `np.fft.fft` para `N=16, 64, 256`.
2. **Convolución rápida**: implementa convolución por FFT y
   compara con `np.convolve` en señales de tamaño 1000.
3. **STFT**: calcula el espectrograma de un chirp y verifica
   que la frecuencia crece linealmente con el tiempo.

## Lecturas recomendadas

- *The Scientist and Engineer's Guide to Digital Signal
  Processing* — Steven W. Smith, libro libre en línea.
- *Fourier Analysis* — Stein & Shakarchi.
- NumPy FFT docs: <https://numpy.org/doc/stable/reference/routines.fft.html>.
- *Understanding Digital Signal Processing* — Lyons.

---

> 📚 **Adaptación al español** de la lección "[Fourier Transform]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Ver [CREDITS.md](../../../../CREDITS.md).
