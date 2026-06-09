# 🎓 Ingeniería de IA desde Cero — versión en español

> **Adaptación al español** del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de **Rohit Ghumare** (`@rohitg00`), publicada bajo la licencia MIT.
> Ver [CREDITS.md](./CREDITS.md) para la atribución completa.

Repositorio oficial del módulo de **Ciencia de Datos y Machine Learning**
del **Diplomado en Machine Learning** de la **Universidad Distrital
Francisco José de Caldas**. Reorganiza el plan del diplomado a un
esquema de **20 fases y ~503 lecciones**, alineado con el currículo
de Rohit Ghumare.

---

## 📌 Aclaración importante

Este repositorio **no es un clon** del repo original. Es una
**adaptación pedagógica al español** que:

1. Reorganiza el plan del Diplomado de la Universidad Distrital en un
   esquema de 20 fases alineado con el currículo de Rohit Ghumare.
2. Adopta la nomenclatura, el plan de estudios y las plantillas del
   repo de Rohit Ghumare (Fase 0 → Fase 19, `docs/es.md`, `code/`,
   `outputs/`, etc.).
3. Mantiene **atribución explícita** al autor original en
   [CREDITS.md](./CREDITS.md), en este README y en la cabecera de
   cada fase.
4. **No duplica** los archivos `docs/en.md` ni el código de las
   lecciones del repo original, en respeto a su `AGENTS.md`.

---

## 🗺️ Las 20 fases

| # | Fase | Slug | Lecciones | Estado |
|:--:|------|------|:---:|:--:|
| 0 | Configuración y herramientas | `00-configuracion-y-herramientas` | 12 | ✅ |
| 1 | Fundamentos de matemáticas | `01-fundamentos-matematicas` | 22 | ✅ |
| 2 | Fundamentos de ML | `02-fundamentos-ml` | 18 | ✅ |
| 3 | Núcleo de Deep Learning | `03-nucleo-deep-learning` | 13 | ⬚ |
| 4 | Visión por computador | `04-vision-por-computador` | 28 | ⬚ |
| 5 | NLP: de fundamentos a avanzado | `05-nlp-fundamentos-a-avanzado` | 29 | ⬚ |
| 6 | Voz y audio | `06-voz-y-audio` | 17 | ⬚ |
| 7 | Transformers a fondo | `07-transformers-a-fondo` | 16 | ⬚ |
| 8 | IA generativa | `08-ia-generativa` | 15 | ⬚ |
| 9 | Aprendizaje por refuerzo | `09-aprendizaje-por-refuerzo` | 12 | ⬚ |
| 10 | LLMs desde cero | `10-llms-desde-cero` | 25 | ⬚ |
| 11 | Ingeniería de LLMs | `11-ingenieria-llms` | 17 | ⬚ |
| 12 | IA multimodal | `12-ia-multimodal` | 25 | ⬚ |
| 13 | Herramientas y protocolos | `13-herramientas-y-protocolos` | 23 | ⬚ |
| 14 | Ingeniería de agentes | `14-ingenieria-agentes` | 42 | ⬚ |
| 15 | Sistemas autónomos | `15-sistemas-autonomos` | — | ⬚ |
| 16 | Multi-agente y enjambres | `16-multi-agente-y-enjambres` | — | ⬚ |
| 17 | Infraestructura y producción | `17-infraestructura-y-produccion` | — | ⬚ |
| 18 | Ética y alineación | `18-etica-y-alineacion` | — | ⬚ |
| 19 | Proyectos capstone | `19-proyectos-capstone` | — | ⬚ |

> ✅ Completa · 🚧 En construcción · ⬚ Pendiente.
> Detalle por fase: [ROADMAP.md](./ROADMAP.md).

---

## 📚 Estructura del repositorio

```text
ia_repository/
├── README.md                  # Este archivo
├── ROADMAP.md                 # Estado por fase
├── CREDITS.md                 # Atribuciones a Rohit Ghumare y Universidad Distrital
├── ESTRUCTURA.md              # Layout completo
├── PLANTILLA_LECCION.md       # Plantilla de lección en español
├── CONTRIBUTING.md           # Cómo contribuir
├── GETTING_STARTED.md         # Guía de inicio rápido
├── LICENSE                    # MIT — Universidad Distrital + crédito a Rohit
├── requirements.txt           # Dependencias
│
├── fases/                     # 20 fases del currículo
│   ├── 00-configuracion-y-herramientas/
│   ├── ...
│   └── 19-proyectos-capstone/
│
├── glosario/                  # Términos canónicos en español
│   └── terminos.md
│
├── datasets/                  # Conjuntos de datos pequeños por fase
├── recursos/                  # Cheatsheets, presentaciones, scripts
└── scripts/                   # Automatización (auditoría, validación, README)
```

Detalles del layout de una lección: [ESTRUCTURA.md](./ESTRUCTURA.md).

---

## 🚀 Cómo usar este repositorio

### Prerrequisitos

- Python 3.10 o superior.
- Git.
- 10 GB de espacio libre (datasets y modelos).
- (Opcional) Docker y una GPU para las fases 3, 4, 9, 10, 11, 12, 14.

### Instalación rápida

```bash
# 1. Clonar
git clone https://github.com/miguelcort/ia_repository.git
cd ia_repository

# 2. Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate     # macOS / Linux
# .venv\Scripts\activate      # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Verificar
python3 -c "import numpy, pandas, sklearn, torch; print('OK')"

# 5. Lanzar Jupyter
jupyter lab
```

Más detalles: [GETTING_STARTED.md](./GETTING_STARTED.md).

### Ruta de aprendizaje sugerida

| Semanas | Foco | Fases |
|---|---|---|
| 1-2 | Setup + matemáticas | Fase 0, Fase 1 |
| 3-4 | ML clásico | Fase 2 |
| 5-6 | Deep Learning | Fase 3 |
| 7-8 | Visión y RL | Fase 4, Fase 9 |
| 9-10 | NLP y Transformers | Fase 5, Fase 7 |
| 11-12 | LLMs | Fase 10, Fase 11 |
| 13-14 | Generativa y multimodal | Fase 8, Fase 12 |
| 15-16 | Agentes y herramientas | Fase 13, Fase 14 |
| 17-18 | Producción y ética | Fase 17, Fase 18 |
| 19-20 | Capstone | Fase 19 |

---

## 🛠️ Tecnologías y herramientas

### Lenguajes y frameworks
- **Python 3.10+** — Lenguaje principal.
- **TypeScript, Rust, Julia** — Para lecciones específicas (siguiendo
  el plan original).

### Librerías núcleo
- **NumPy, Pandas, SciPy** — Computación científica.
- **scikit-learn, XGBoost, LightGBM** — ML clásico.
- **PyTorch, TensorFlow/Keras** — Deep Learning.
- **OpenCV, Pillow** — Visión por computador.
- **Gymnasium** — Aprendizaje por refuerzo.
- **Hugging Face Transformers / Datasets / PEFT** — LLMs.

### MLOps y despliegue
- **Docker** — Contenerización.
- **MLflow, DVC** — Tracking y versionamiento.
- **FastAPI** — APIs de modelos.
- **Optuna** — Ajuste de hiperparámetros.
- **SHAP, LIME** — Explicabilidad.

---

## 🤝 Cómo contribuir

¡Las contribuciones son bienvenidas! Por favor lee
[CONTRIBUTING.md](./CONTRIBUTING.md) y la
[PLANTILLA_LECCION.md](./PLANTILLA_LECCION.md) antes de abrir un PR.

Reglas resumidas:

- **Una lección = un commit.** Asunto: `feat(fase-NN/MM): <slug>`.
- Documentación en `docs/es.md`, código en `code/main.<lang>`,
  tests con 5+ casos en `code/tests/`.
- Diagramas en Mermaid o SVG (nunca con caracteres Unicode).
- No copies código de otros repos de currículo; cita papers, RFCs y
  documentación oficial cuando sea el caso.

---

## 📂 Material heredado

El material se construye lección por lección dentro de cada fase bajo
`fases/NN-nombre/MM-slug/`. La estructura del repositorio se documenta
en [ESTRUCTURA.md](./ESTRUCTURA.md).

---

## 🙏 Agradecimientos y atribuciones

Este repositorio existe gracias a dos contribuciones:

### 1. Plan de estudios original — **Rohit Ghumare**

El currículo [**AI Engineering from
Scratch**](https://github.com/rohitg00/ai-engineering-from-scratch) de
**Rohit Ghumare** ([github.com/rohitg00](https://github.com/rohitg00))
es la base pedagógica de este proyecto. 503 lecciones, 20 fases,
cuatro lenguajes (Python, TypeScript, Rust, Julia), licencia MIT.

- Repositorio: <https://github.com/rohitg00/ai-engineering-from-scratch>
- Sitio web: <https://aiengineeringfromscratch.com>
- Otro proyecto destacado del autor: [Agent Memory — #1 Persistent
  Memory ⭐](https://github.com/rohitg00/agentmemory).

> **Gracias, Rohit, por hacer accesible un plan completo de
> ingeniería de IA bajo MIT.** Este trabajo en español no sería
> posible sin tu generosidad.

### 2. Adaptación al español — **Miguel Cortés** y la **Universidad Distrital**

- **Adaptación, migración y traducción:** Miguel Cortés
  ([@miguelcort](https://github.com/miguelcort)).
- **Institución académica:** [Universidad Distrital Francisco José de
  Caldas](https://www.udistrital.edu.co) — Diplomado en Machine Learning.
- **Estudiantes y profesores** del Diplomado de la Universidad Distrital
  que han iterado sobre el material y han inspirado la reorganización.

La atribución detallada, con instrucciones de citación BibTeX y la
relación de licencias, está en [CREDITS.md](./CREDITS.md).

---

## 📝 Licencia

- **Este repositorio:** MIT — ver [LICENSE](./LICENSE).
- **Repositorio original (AI Engineering from Scratch):** MIT —
  Copyright (c) Rohit Ghumare.

El texto del MIT exige preservar el aviso de copyright y el permiso
en todas las copias. Este README y [CREDITS.md](./CREDITS.md) cumplen
con ese requisito.

---

## 📧 Contacto

- **Repositorio (issues, PRs):** <https://github.com/miguelcort/ia_repository>
- **Repositorio original (issues, PRs):** <https://github.com/rohitg00/ai-engineering-from-scratch>
- **Sitio web del autor original:** <https://aiengineeringfromscratch.com>
- **Universidad Distrital:** <https://www.udistrital.edu.co>

---

## ⭐ Una última cosa

Si este material te resulta útil, considera:

- Darle una estrella al [repositorio
  original](https://github.com/rohitg00/ai-engineering-from-scratch)
  de Rohit Ghumare.
- Darle una estrella a [este repositorio](https://github.com/miguelcort/ia_repository).
- Contribuir con una nueva lección, traducción o corrección.

**¡Éxitos en tu viaje por el mundo de la ingeniería de IA!** 🚀
