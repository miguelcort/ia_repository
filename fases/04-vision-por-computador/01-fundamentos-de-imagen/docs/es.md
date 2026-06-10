# Fundamentos de imagen

> Antes de una red neuronal, una imagen es solo una matriz de numeros. Entender la representacion (uint8, float, RGB, escala de grises, resize) es prerequisito para todo lo demas.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 03-nucleo-deep-learning/10-mini-framework
**Tiempo estimado:** ~25 minutos

## Objetivos de aprendizaje

- Crear imagenes de prueba (rayas, color).
- Convertir RGB a grises.
- Normalizar uint8 a float [0,1] o [-1,1].
- Redimensionar con interpolacion bilineal.

## Constrúyelo

```python
def normalizar_imagen(img, modo="01"):
    f = img.astype(np.float32) / 255.0
    if modo == "11":
        f = f * 2.0 - 1.0
    return f
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-image-prep
fase: 04
leccion: 01
---

1. Clasificacion: 224x224, /255, stats ImageNet.
2. Segmentacion: resize + pad, sin stats ImageNet.
3. Deteccion: multiplo de 32 (640x640).
4. Mismo preprocesamiento train/val.
```

## Ejercicios

1. **Data augmentation**: implementa random flip, crop,
   color jitter.
2. **Histograma**: visualiza distribucion de intensidades
   por canal.
3. **Desafio**: implementa una clase Dataset que cargue,
   redimensione y normalice imagenes desde un directorio.

## Lecturas recomendadas

- "Computer Vision: Algorithms and Applications" (Szeliski)
- PIL/Pillow: <https://pillow.readthedocs.io/>
- torchvision transforms: <https://pytorch.org/vision/stable/transforms.html>

---

> 📚 **Adaptación al español** de la lección "[Image Fundamentals]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).