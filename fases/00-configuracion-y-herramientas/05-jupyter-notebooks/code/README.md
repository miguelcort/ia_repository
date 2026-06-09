# Jupyter Notebooks

Valida la convencion del diplomado para un notebook:

```bash
cd code
python3 main.py mi_notebook.ipynb
```

Ejecuta los tests:

```bash
python3 -m unittest discover -s tests -v
```

O desde la raíz del repo:

```bash
python3 scripts/probar_lecciones.py --quiet
```
