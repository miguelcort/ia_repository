# Entorno de desarrollo

Ejecuta el demo y observa el reporte JSON de tu entorno:

```bash
cd code
python3 main.py
```

Ejecuta los tests:

```bash
python3 -m unittest discover -s tests -v
```

O desde la raíz del repo:

```bash
python3 scripts/probar_lecciones.py --quiet
```

Archivar el entorno en un archivo (útil para reproducibilidad):

```bash
python3 code/main.py > entorno.json
```
