# 02 — Reward hacking y Goodhart

Ejecuta el demo de curvas de sobre-optimización:

```bash
cd code
python3 main.py
```

Deberías ver:

- Trayectorias KL-proxy-oro para proxies ajustados con 100, 300,
  1000 muestras.
- Comparación Gaussiana vs Student-t (cola pesada) que muestra
  *Catastrophic Goodhart*.

Ejecuta los tests:

```bash
python3 -m unittest discover -s tests -v
```

Las 7 clases de tests cubren: gold reward, fit del proxy RM,
solución lineal, divergencia KL, hill-climb y Student-t.
