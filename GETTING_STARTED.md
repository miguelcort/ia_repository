# 🚀 Guía de Inicio Rápido

Bienvenido al currículo de **Ingeniería de IA en español** mantenido
por **Miguel Cortés** ([@miguelcort](https://github.com/miguelcort)),
basado en [AI Engineering from
Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) de
**Rohit Ghumare**.

> Antes de empezar, revisa [CREDITS.md](./CREDITS.md) y
> [ROADMAP.md](./ROADMAP.md) para entender el plan de estudios.

---

## 📋 Antes de empezar

### Requisitos previos

- **Python 3.10 o superior** instalado.
- **Git** instalado y configurado.
- Conocimientos básicos de programación en Python.
- 10 GB de espacio libre en disco (datasets, modelos, entornos).

### Conocimientos recomendados (no bloqueantes)

- Álgebra lineal básica y estadística.
- Manejo de la línea de comandos.
- Inglés para leer documentación técnica.

---

## ⚡ Instalación rápida

### 1. Clonar el repositorio

```bash
git clone https://github.com/miguelcort/ia_repository.git
cd ia_repository
```

### 2. Crear un entorno virtual

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell):**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verificar la instalación

```bash
python3 -c "import numpy, pandas, sklearn, torch, transformers; print('OK')"
```

### 5. Lanzar Jupyter

```bash
jupyter lab
```

---

## 🗺️ Tu primera lección

1. Lee el [README principal](./README.md) y este archivo.
2. Abre el [ROADMAP.md](./ROADMAP.md) para ver el estado de las 20
   fases.
3. Ve a la fase que te corresponda según tu nivel (recomendamos
   empezar por la [Fase 0 — Configuración y
   herramientas](./fases/00-configuracion-y-herramientas/README.md) o
   la [Fase 2 — Fundamentos de
   ML](./fases/02-fundamentos-ml/README.md) si ya programas).
4. Cada lección vive en
   `fases/NN-nombre-fase/MM-slug-leccion/` con esta estructura:

```text
   MM-slug/
   ├── docs/es.md         # Lee esto primero
   ├── code/main.py       # Ejecútalo: python3 main.py
   ├── code/tests/        # Pasa los tests
   └── notebooks/         # Experimentación opcional
```

5. Ejecuta la lección:

   ```bash
   cd fases/02-fundamentos-ml/02-regresion-lineal-desde-cero/code
   python3 main.py
   python3 -m unittest discover tests -v
```

---

## 📚 Ruta de aprendizaje sugerida

| Semanas | Foco | Fases |
|---|---|---|
| 1-2 | Setup y matemáticas | 0, 1 |
| 3-4 | ML clásico | 2 |
| 5-6 | Deep Learning y visión | 3, 4 |
| 7-8 | NLP y RL | 5, 9 |
| 9-10 | Transformers y LLMs | 7, 10 |
| 11-12 | Ingeniería de LLMs y multimodal | 11, 12 |
| 13-14 | Generativa, herramientas y agentes | 8, 13, 14 |
| 15-16 | Producción y ética | 17, 18 |
| 17-18 | Multi-agente y sistemas autónomos | 15, 16 |
| 19-20 | Capstone | 19 |

---

## 🛠️ Herramientas recomendadas

### Editores

- **VS Code** + extensiones *Python*, *Jupyter*, *Pylance*, *GitLens*.
- **PyCharm Community** como IDE completo.
- **JupyterLab** para exploración interactiva.

### Línea de comandos

- **zsh** (macOS por defecto) o **bash** (Linux).
- **Git** con configuración de `user.name` y `user.email`.

### Opcionales

- **Docker** para las fases 17 y superiores.
- **GPU NVIDIA** para entrenar modelos pequeños (CUDA 12+).
- **W&B** o **MLflow** para tracking de experimentos.

---

## 🆘 Solución de problemas comunes

### "ModuleNotFoundError: No module named 'X'"

```bash
# Verifica que el entorno virtual esté activo
which python
# Debe apuntar a .venv/bin/python

# Reinstala la dependencia
pip install X
```

### "jupyter: command not found"

```bash
pip install --upgrade jupyter jupyterlab
python -m ipykernel install --user --name=venv
```

### Los notebooks no encuentran los módulos del repositorio

```bash
# Instala el repo en modo editable
pip install -e .
```

### Errores de PyTorch en macOS Apple Silicon

```bash
# PyTorch instala el backend MPS automáticamente; no requiere CUDA
python3 -c "import torch; print(torch.backends.mps.is_available())"
```

### Quiero entrenar un modelo grande pero no tengo GPU

- Usa **Google Colab** o **Kaggle Notebooks** (ambos con GPU gratuita).
- Monta este repositorio en Colab con:

  ```python
  !git clone https://github.com/miguelcort/ia_repository.git
```

---

## 📞 Soporte

### Recursos de ayuda

- [Issues del repositorio](https://github.com/miguelcort/ia_repository/issues)
- Foros de la comunidad y canales del autor original.
- Documentación oficial de cada librería (enlaces en
  [`recursos/`](./recursos/)).

### Canales del proyecto original

- Repositorio de Rohit Ghumare:
  <https://github.com/rohitg00/ai-engineering-from-scratch/issues>
- Sitio web: <https://aiengineeringfromscratch.com>

---

## ✅ Checklist de inicio

- [ ] Python 3.10+ instalado y funcionando.
- [ ] Repositorio clonado.
- [ ] Entorno virtual creado y activado.
- [ ] Dependencias instaladas sin errores.
- [ ] JupyterLab lanzando correctamente.
- [ ] Leí [CREDITS.md](./CREDITS.md) y [ROADMAP.md](./ROADMAP.md).
- [ ] Ejecuté mi primera lección (`python3 main.py` exit 0).
- [ ] Pasé los tests de mi primera lección.
- [ ] Configuré mi editor favorito.

---

## 📈 Próximos pasos

1. Lee la documentación de la fase que vas a empezar.
2. Ejecuta el `main.py` de la primera lección.
3. Lee el `docs/es.md` correspondiente.
4. Resuelve los ejercicios propuestos.
5. Cuando te sientas cómodo, envía un PR con mejoras.

---

## ✨ ¡Estás listo!

Ya tienes todo lo necesario para empezar tu viaje de **20 fases** por
la ingeniería de IA. La clave es la práctica constante: ejecuta el
código, modifica los ejemplos, rompe cosas y vuelve a armar.

**¡Adelante y mucho éxito!** 🚀

---

**Última actualización:** Junio 2026
**Versión:** 2.0 (migración al esquema de 20 fases)
