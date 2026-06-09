# Docker para IA

El `Dockerfile` está en este directorio. Construye la imagen:

```bash
cd code
docker build -t leccion-07 .
docker run --rm leccion-07
```

Para iterar sin rebuild (monta el código como volumen):

```bash
docker run --rm -v "$PWD":/app leccion-07 python /app/code/main.py
```

Ejecuta los tests (validan la sintaxis del Dockerfile sin Docker):

```bash
python3 -m unittest discover -s tests -v
```

O desde la raíz del repo:

```bash
python3 scripts/probar_lecciones.py --quiet
```
