# Numeros complejos para IA

> Los complejos no son abstractos: aparecen en rotaciones, FFT y en el procesamiento moderno de senales.

**Tipo:** Aprender
**Lenguajes:** Python
**Prerrequisitos:** 01-intuicion-algebra-lineal
**Tiempo estimado:** ~30 minutos

## Objetivos de aprendizaje

- Implementar aritmetica con numeros complejos.
- Aplicar la formula de Euler.
- Calcular FFT y entender su uso en senales.

## Constrúyelo

```python
import cmath
import math

def multiplicar(a, b):
    real = a[0] * b[0] - a[1] * b[1]
    imag = a[0] * b[1] + a[1] * b[0]
    return (real, imag)

def modulo(a):
    return math.sqrt(a[0] ** 2 + a[1] ** 2)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-fft-uso
fase: 01
leccion: 19
---

1. Para senales largas, FFT.
2. Para streaming, STFT.
3. Para convoluciones, FFT-based.
```

## Ejercicios

1. **Convolucion FFT**: implementa la convolucion 1D usando FFT.
2. **Espectrograma**: visualiza el spectrograma de una senal de audio.
3. **Desafio**: implementa una red convolucional con valores complejos.

## Lecturas recomendadas

- "Digital Signal Processing" (Proakis & Manolakis)
- numpy.fft: <https://numpy.org/doc/stable/reference/routines.fft.html>

---

> 📚 **Adaptación al español** de la lección "[Complex Numbers]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).