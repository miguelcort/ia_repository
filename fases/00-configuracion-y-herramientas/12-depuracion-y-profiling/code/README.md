# Depuracion y profiling

Micro-benchmark y perfilado de una funcion lenta:

```bash
cd code
python3 main.py --modo benchmark --n 100
python3 main.py --modo profile --n 100
python3 main.py --modo memory --n 100
python3 main.py --modo debug
```

Ejecuta los tests:

```bash
python3 -m unittest discover -s tests -v
```

O desde la raíz del repo:

```bash
python3 scripts/probar_lecciones.py --quiet
```
