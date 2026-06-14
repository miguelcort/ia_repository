# Fase 0 — Configuración y herramientas

> Prepara el entorno de trabajo para todo lo que viene después.

Esta fase es la **puerta de entrada** al currículo. Antes de escribir la
primera línea de NumPy, la primera red neuronal o el primer agente,
necesitamos que la máquina del estudiante sea un instrumento afinado:
mismo Python, mismo gestor de paquetes, misma shell, mismo editor, mismo
flujo de Git. Sin ese suelo común, las 19 fases que siguen se convierten
en una pelea constante contra `ImportError`, versiones rotas y entornos
contaminados.

La filosofía de la fase es **bottom-up**: instalamos primero el sistema
base, luego los gestores de paquetes, luego los lenguajes, y por último
las bibliotecas de IA. Cada capa depende estrictamente de la de abajo;
si una capa falla, nada de lo que pongamos encima funcionará de forma
confiable. Por eso dedicamos una lección entera a verificar el entorno
y dejamos el resultado como un artefacto reutilizable (un JSON firmable
y comparable) que el estudiante puede archivar junto a cada proyecto.

## Índice de lecciones

| # | Lección | Tipo | Resumen |
|--:|---|---|---|
| 01 | [Entorno de desarrollo](01-entorno-desarrollo/) | Construir | Verificador JSON de capas (intérprete, paquetes, GPU). |
| 02 | [Git y colaboración](02-git-colaboracion/) | Aprender | Flujo de trabajo con `git init`, ramas, PRs, resolución de conflictos. |
| 03 | [Configuración de GPU y nube](03-gpu-nube/) | Construir | Detección de CUDA, MPS y alternativas en la nube (Colab, Lambda, Vast). |
| 04 | [APIs y claves](04-apis-y-claves/) | Construir | Manejo seguro de tokens, archivos `.env` y rotación. |
| 05 | [Jupyter Notebooks](05-jupyter-notebooks/) | Construir | Notebooks reproducibles: kernels, magic commands y exportación. |
| 06 | [Entornos virtuales de Python](06-entornos-python/) | Construir | `uv`, `venv` y `conda` comparados con un mismo proyecto. |
| 07 | [Docker para IA](07-docker-para-ia/) | Construir | Imágenes con CUDA, `docker compose` y volúmenes para datasets. |
| 08 | [Configuración del editor (VS Code)](08-configuracion-editor/) | Construir | Extensiones, `settings.json` y atajos para ingeniería de IA. |
| 09 | [Gestión de datos](09-gestion-de-datos/) | Construir | Versionado de datos con DVC y layouts reproducibles. |
| 10 | [Terminal y shell](10-terminal-y-shell/) | Aprender | `bash`/`zsh` productivos: pipes, alias, `rg`, `fzf`. |
| 11 | [Linux para IA](11-linux-para-ia/) | Aprender | Permisos, `systemd`, `htop`, drivers NVIDIA. |
| 12 | [Depuración y profiling](12-depuracion-y-profiling/) | Construir | `pdb`, `cProfile`, `py-spy` y trazado de cuellos de botella. |

## Prerrequisitos

- Ninguno. Esta fase es el punto de partida del currículo.
- Conocimiento básico de la terminal del sistema operativo.
- Una conexión a internet estable para descargar paquetes.

## Objetivos de la fase

Al terminar esta fase el estudiante podrá:

- **Diagnosticar** las cinco capas del entorno (sistema, drivers,
  intérprete, paquetes, frameworks) con un verificador determinista.
- **Configurar** entornos virtuales reproducibles con `uv` o `conda`.
- **Versionar** código y notebooks con Git, y datasets con DVC.
- **Contenerizar** un experimento reproducible con Docker.
- **Depurar** cuellos de botella con `cProfile`, `py-spy` y el profiler
  nativo de VS Code.
- **Proteger** claves de API y secretos con archivos `.env` y
  variables de entorno.
- **Elegir** entre entrenamiento local y en la nube según el costo
  y la urgencia.

## Stack y herramientas

- **Lenguaje principal:** Python 3.10+ (recomendado 3.12).
- **Gestor de paquetes:** `uv` (10–100x más rápido que `pip`).
- **Entornos aislados:** `uv venv` o `conda`.
- **Control de versiones:** Git 2.40+ con `git-lfs` opcional.
- **Contenedores:** Docker 24+ con soporte para `nvidia-container-toolkit`.
- **Editor:** VS Code con las extensiones *Python*, *Pylance*,
  *Jupyter* y *Docker*.
- **Notebooks:** JupyterLab 4.x o VS Code con el kernel de Python.
- **GPU (opcional):** CUDA 12.x para tarjetas NVIDIA; MPS para
  Apple Silicon.

## Conceptos clave

| Concepto | Por qué importa |
|---|---|
| **Capas del entorno** | Cada capa (sistema → drivers → intérprete → paquetes → frameworks) puede fallar de forma independiente. |
| **Entorno virtual** | Aísla versiones de paquetes por proyecto. Sin él, actualizar PyTorch para un proyecto rompe otro. |
| **Reproducibilidad** | Si tu `requirements.txt` no congela versiones, el experimento de hoy no funcionará mañana. |
| **Hash de entorno** | El verificador de la lección 01 produce un JSON firmable que detecta *deriva* entre ejecuciones. |
| **Sandbox de GPU** | Docker con `nvidia-container-toolkit` aísla el entrenamiento de tu host y facilita compartir el ambiente. |

## Cómo estudiar esta fase

1. **Empieza por la lección 01** (verificador de entorno). Aunque sea
   aburrida, te ahorrará horas en las fases siguientes.
2. **Avanza en orden numérico.** Cada lección construye sobre la
   anterior: la 06 (entornos) asume que tienes Python de la 01; la
   07 (Docker) asume la 06.
3. **Ejecuta cada demo localmente.** Si `python3 main.py` no termina
   con código 0, **no avances** a la siguiente lección.
4. **Archiva el JSON del verificador** en tu proyecto. Te servirá
   cuando algo se rompa tres meses después.
5. **Salta la lección 11** (Linux) si ya eres usuario intermedio; vuelve
   a ella cuando necesites configurar drivers NVIDIA.

## Verificación de progreso

```bash
# Lección 01 — corre el verificador
python3 fases/00-configuracion-y-herramientas/01-entorno-desarrollo/code/main.py > entorno.json

# Lección 06 — crea y exporta un entorno reproducible
uv venv && source .venv/bin/activate
uv pip install numpy jupyter pytest

# Lección 07 — contenedor funcional
docker run --rm --gpus all hello-nvidia
```

Si las tres verificaciones pasan en menos de 30 minutos, el estudiante
está listo para saltar a la Fase 1.

## Conexión con otras fases

- **Salida natural** → [Fase 1 — Fundamentos de matemáticas](../01-fundamentos-matematicas/README.md)
  (las primeras lecciones usan NumPy y álgebra lineal).
- **Requisitos de retorno** → las fases 17 y 19 vuelven sobre Docker
  y depuración para MLOps y proyectos capstone.
- **Material complementario** → [glosario/terminos.md](../../glosario/terminos.md)
  define *entorno virtual*, *kernel*, *driver* y *worktree*.

## Véase también

- [ROADMAP.md](../../ROADMAP.md) — estado de las 20 fases.
- [PLANTILLA_LECCION.md](../../PLANTILLA_LECCION.md) — cómo se
  estructura cada lección.
- [GETTING_STARTED.md](../../GETTING_STARTED.md) — instalación rápida.
- [ESTRUCTURA.md](../../ESTRUCTURA.md) — layout general del repositorio.
- [requirements.txt](../../requirements.txt) — dependencias consolidadas.

---

> 📚 **Adaptación al español** del plan de estudios
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](../../CREDITS.md).
