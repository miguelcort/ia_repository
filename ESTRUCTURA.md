# 📐 Estructura del Repositorio

> **Adaptación al español** del esquema
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de Rohit Ghumare (MIT). Ver [CREDITS.md](./CREDITS.md).

Este documento describe la organización de carpetas. Cualquier aporte
nuevo debe seguir este mismo layout.

---

## Vista de alto nivel

```text
ia_repository/
│
├── README.md                         # Cara pública del proyecto (en español)
├── ROADMAP.md                        # Estado de las 20 fases
├── CREDITS.md                        # Atribuciones y licencias
├── CONTRIBUTING.md                   # Cómo contribuir
├── LICENSE                           # MIT — Universidad Distrital + crédito a Rohit
├── GETTING_STARTED.md                # Guía rápida de instalación
├── PLANTILLA_LECCION.md              # Plantilla canónica de una lección
├── ESTRUCTURA.md                     # Este archivo
├── requirements.txt                  # Dependencias consolidadas
│
├── fases/                            # 20 fases del currículo
│   ├── 00-configuracion-y-herramientas/
│   ├── 01-fundamentos-matematicas/
│   ├── 02-fundamentos-ml/            # ← proviene de Clase 1, 2 y 3
│   ├── 03-nucleo-deep-learning/      # ← proviene de Clase 4
│   ├── 04-vision-por-computador/     # ← proviene de Clase 5
│   ├── 05-nlp-fundamentos-a-avanzado/
│   ├── 06-voz-y-audio/
│   ├── 07-transformers-a-fondo/
│   ├── 08-ia-generativa/
│   ├── 09-aprendizaje-por-refuerzo/  # ← proviene de Clase 6
│   ├── 10-llms-desde-cero/
│   ├── 11-ingenieria-llms/
│   ├── 12-ia-multimodal/
│   ├── 13-herramientas-y-protocolos/
│   ├── 14-ingenieria-agentes/
│   ├── 15-sistemas-autonomos/
│   ├── 16-multi-agente-y-enjambres/
│   ├── 17-infraestructura-y-produccion/   # MLOps y despliegue
│   ├── 18-etica-y-alineacion/        # IA responsable
│   └── 19-proyectos-capstone/        # Proyecto final integrador
│
├── glosario/                         # Términos canónicos del plan
│   └── terminos.md
│
├── datasets/                         # Datasets pequeños de ejemplo por fase
│
├── recursos/                         # Cheatsheets, presentaciones, scripts auxiliares
│
├── scripts/                          # Automatización (auditoría, README, glosario)
│
└── .github/workflows/                # CI: auditoría, conteo de lecciones
```

---

## Estructura de una fase

```text
fases/NN-nombre-de-la-fase/
├── README.md                         # Objetivos, prerrequisitos, índice de lecciones
└── NN-slug-de-leccion/               # Lección individual
    ├── docs/
    │   └── es.md                     # Narrativa en español (obligatorio)
    ├── code/
    │   ├── main.py                   # Implementación ejecutable
    │   ├── README.md                 # Cómo ejecutar
    │   └── tests/
    │       └── test_main.py          # Mínimo 5 pruebas unitarias
    ├── notebooks/                    # Opcional: exploración Jupyter
    │   └── NN_nombre.ipynb
    ├── ejercicios/                   # Opcional: ejercicios propuestos
    │   └── README.md
    ├── referencias/                  # Opcional: lecturas externas
    │   └── recursos.md
    └── outputs/                      # Artefacto reutilizable (prompt, skill, agente, MCP)
        └── skill-<slug>.md
```

---

## Anatomía de una lección

Cada lección sigue la secuencia **PROBLEMA → CONCEPTO → CONSTRUIR → USAR → DESPLEGAR**,
adaptada al español. Ver [PLANTILLA_LECCION.md](./PLANTILLA_LECCION.md) para el detalle.

```text
PROBLEMA     dolor concreto, por qué importa
   ↓
CONCEPTO     intuición, diagramas, matemática mínima
   ↓
CONSTRUIR    implementación desde cero (Python, TypeScript, Rust o Julia)
   ↓
USAR         mismo algoritmo con la librería de producción
   ↓
DESPLEGAR    prompt, skill, agente o servidor MCP reutilizable
```

---

## Reglas duras

1. **Una lección = una carpeta.** Nunca agrupar varias lecciones en la misma carpeta.
2. **Un commit por lección.** Cada `git commit` debe tocar **una** carpeta de lección.
3. **Idioma de la documentación:** español (`docs/es.md`). El código va en
   inglés por convención.
4. **Diagramas:** Mermaid o SVG. No usar caracteres Unicode para dibujar
   diagramas.
5. **Bloques de código con etiqueta de lenguaje.** Siempre:
   `python`, `typescript`, `rust`, `julia`, `bash`, `json`, `mermaid`, `yaml`, etc.
6. **Dependencias permitidas** (ver `requirements.txt` raíz):
   NumPy, Pandas, SciPy, scikit-learn, XGBoost, LightGBM, TensorFlow,
   PyTorch, OpenCV, Gymnasium, MLflow, FastAPI, Optuna, Jupyter, pytest.
7. **No committear archivos generados.** El `.gitignore` raíz cubre
   `__pycache__/`, datasets grandes, checkpoints, etc.

---

## Cómo encaja con la Universidad Distrital

El Diplomado se entrega como un currículo continuo de 20 fases. Los
bloques temáticos tradicionales se mapean así:

| Bloque temático | Fases destino |
|---|---|
| Introducción a ML | Fases 0 y 2 |
| Procesamiento de datos | Fase 2 |
| Modelos estadísticos | Fase 2 |
| Introducción a Deep Learning | Fase 3 |
| Convoluciones y segmentación | Fase 4 |
| Aprendizaje por refuerzo | Fase 9 |
| Ética en IA | Fase 18 |
| Docker y MLOps | Fase 17 |
| Prueba técnica final | Fase 19 |

---

## Véase también

- [PLANTILLA_LECCION.md](./PLANTILLA_LECCION.md) — Cómo escribir una lección.
- [CONTRIBUTING.md](./CONTRIBUTING.md) — Cómo enviar un PR.
- [ROADMAP.md](./ROADMAP.md) — Estado actual de cada fase.
- [glosario/terminos.md](./glosario/terminos.md) — Vocabulario canónico.
