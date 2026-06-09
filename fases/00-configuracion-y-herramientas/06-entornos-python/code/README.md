# Entornos virtuales de Python

Gestor de venv portable con API de comandos:

```bash
cd code
chmod +x main.sh
NOMBRE_ENTORNO=mi-env ./main.sh create
NOMBRE_ENTORNO=mi-env ./main.sh check
NOMBRE_ENTORNO=mi-env ./main.sh python -c "import sys; print(sys.prefix)"
NOMBRE_ENTORNO=mi-env ./main.sh clean
```

Ejecuta los tests:

```bash
bash code/tests/test_main.sh
```

O desde la raíz del repo:

```bash
python3 scripts/probar_lecciones.py --quiet
```
