# 🗺️ Roadmap — Ingeniería de IA desde Cero (en español)

> **Adaptación al español** del plan de estudios
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> de **Rohit Ghumare** (MIT). Ver [CREDITS.md](./CREDITS.md).

Este roadmap refleja el estado real del repositorio dentro del esquema
de 20 fases. Cada fase puede tener tres estados:

| Glifo | Estado | Significado |
|:---:|---|---|
| ✅ | **Completa** | Carpeta creada, `README.md`, `docs/es.md` y al menos un esqueleto de lección con `code/`. |
| 🚧 | **En construcción** | Carpeta y README listos; las lecciones aún no tienen código. |
| ⬚ | **Pendiente** | Carpeta no creada todavía; sólo aparece en este roadmap. |

---

## Visión general

```text
Fase 0   ─ Setup y herramientas
Fase 1   ─ Fundamentos de matemáticas
Fase 2   ─ Fundamentos de ML clásico
Fase 3   ─ Núcleo de Deep Learning
Fase 4   ─ Visión por computador
Fase 5   ─ NLP: de fundamentos a avanzado
Fase 6   ─ Voz y audio
Fase 7   ─ Transformers a fondo
Fase 8   ─ IA generativa
Fase 9   ─ Aprendizaje por refuerzo
Fase 10  ─ LLMs desde cero
Fase 11  ─ Ingeniería de LLMs en producción
Fase 12  ─ IA multimodal
Fase 13  ─ Herramientas y protocolos
Fase 14  ─ Ingeniería de agentes
Fase 15  ─ Sistemas autónomos
Fase 16  ─ Multi-agente y enjambres
Fase 17  ─ Infraestructura y producción
Fase 18  ─ Ética y alineación
Fase 19  ─ Proyectos capstone
```

---

## Estado por fase

| # | Fase | Slug | Lecciones (plan) | Estado |
|:--:|------|------|:---:|:--:|
| 0 | Setup y herramientas | `00-configuracion-y-herramientas` | 12 | ✅ |
| 1 | Fundamentos de matemáticas | `01-fundamentos-matematicas` | 22 | ✅ |
| 2 | Fundamentos de ML | `02-fundamentos-ml` | 18 | ✅ |
| 3 | Núcleo de Deep Learning | `03-nucleo-deep-learning` | 13 | ✅ |
| 4 | Visión por computador | `04-vision-por-computador` | 28 | ✅ |
| 5 | NLP | `05-nlp-fundamentos-a-avanzado` | 29 | ✅ |
| 6 | Voz y audio | `06-voz-y-audio` | 17 | ✅ |
| 7 | Transformers a fondo | `07-transformers-a-fondo` | 16 | ✅ |
| 8 | IA generativa | `08-ia-generativa` | 15 | ✅ |
| 9 | Aprendizaje por refuerzo | `09-aprendizaje-por-refuerzo` | 12 | ✅ |
| 10 | LLMs desde cero | `10-llms-desde-cero` | 25 | 🚧 |
| 11 | Ingeniería de LLMs | `11-ingenieria-llms` | 17 | ✅ |
| 12 | IA multimodal | `12-ia-multimodal` | 25 | ✅ |
| 13 | Herramientas y protocolos | `13-herramientas-y-protocolos` | 23 | ✅ |
| 14 | Ingeniería de agentes | `14-ingenieria-agentes` | 42 | 🚧 |
| 15 | Sistemas autónomos | `15-sistemas-autonomos` | — | 🚧 |
| 16 | Multi-agente y enjambres | `16-multi-agente-y-enjambres` | — | ⬚ |
| 17 | Infraestructura y producción | `17-infraestructura-y-produccion` | — | ⬚ |
| 18 | Ética y alineación | `18-etica-y-alineacion` | — | ⬚ |
| 19 | Proyectos capstone | `19-proyectos-capstone` | — | ⬚ |

El conteo exacto de lecciones por fase lo calcula automáticamente
`scripts/actualizar_conteo.py` a partir de las carpetas creadas.

---

## Política de commits y PRs

Ver [CONTRIBUTING.md](./CONTRIBUTING.md). Reglas resumen:

- **Un commit por lección.** Nunca agrupar varias lecciones en un solo commit.
- Asunto de commit `<72` caracteres: `feat(fase-NN/MM): <slug>`.
- Toda lección debe tener `docs/es.md`, `code/main.<lang>` y
  `code/tests/test_main.*` con 5+ pruebas.

---

## Referencias

- [CREDITS.md](./CREDITS.md) — Atribuciones completas.
- [ESTRUCTURA.md](./ESTRUCTURA.md) — Layout del repositorio.
- [PLANTILLA_LECCION.md](./PLANTILLA_LECCION.md) — Plantilla de lección.
- Repositorio original:
  <https://github.com/rohitg00/ai-engineering-from-scratch>
