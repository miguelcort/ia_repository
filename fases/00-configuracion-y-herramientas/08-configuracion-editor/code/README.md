# Configuracion del editor (VS Code)

Verifica que tu `.vscode/` cumple la convencion del currículo:

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
