# La transformada de Fourier

> La FFT es uno de los algoritmos mas importantes del siglo XX. Convierte senales en el dominio del tiempo al de la frecuencia.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 19-numeros-complejos
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Aplicar FFT y FFT inversa a una senal.
- Identificar picos en el espectro.
- Implementar convolucion por FFT.
- Diagnosticar aliasing y Nyquist.

## Constrúyelo

```python
import numpy as np

def fft(senal):
    return np.fft.fft(senal)

def fft_frecuencias(N, fs=1.0):
    return np.fft.fftfreq(N, d=1 / fs)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-fft-decide
fase: 01
leccion: 20
---

1. Estacionarias: FFT.
2. No estacionarias: STFT o wavelet.
3. Limpiar ruido: filtrar en frecuencia.
4. Convoluciones grandes: FFT conv.
```

## Ejercicios

1. **STFT**: implementa short-time Fourier transform.
2. **Filtro pasa-bajos**: aplica FFT, elimina frecuencias altas,
   aplica IFFT.
3. **Desafio**: implementa un vocoder (analisis-sintesis FFT).

## Lecturas recomendadas

- "Digital Signal Processing" (Proakis & Manolakis)
- numpy.fft: <https://numpy.org/doc/stable/reference/routines.fft.html>

---

> 📚 **Adaptación al español** de la lección "[Fourier Transform]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).