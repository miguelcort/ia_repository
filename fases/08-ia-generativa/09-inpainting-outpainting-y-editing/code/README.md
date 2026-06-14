# 09 — Inpainting, outpainting y edición de imagen

Ejecuta el demo del inpainter DDPM de juguete en 5-D:

```bash
cd code
python3 main.py
```

Deberías ver:

- Entrenamiento del DDPM sobre la mezcla de dos clústeres 5-D.
- Inpainting: pinea dims 0-2 y regenera dims 3-4 — los valores
  regenerados deben *matchear* el signo del clúster de las dims
  pineadas.
- Outpainting: enmascara dims 0-1, pinea dims 2-4, regenera la
  "cabeza".

Ejecuta los tests:

```bash
python3 -m unittest discover -s tests -v
```

Las 8 clases de tests cubren: embeddings sinusoidales, helpers
matemáticos, schedule de ruido, muestreo de datos, forward pass
de la red, inpainting (respeto de dims pineadas), main.
