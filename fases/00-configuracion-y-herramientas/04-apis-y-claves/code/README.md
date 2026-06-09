# APIs y claves

Carga y valida claves desde `.env` y variables de entorno:

```bash
cd code
cp .env.example .env  # editar con tus claves
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
