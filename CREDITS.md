# Créditos y Atribuciones

Este repositorio (`ia_repository`) es una **adaptación al español** del currículo
**[AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)**
de **Rohit Ghumare** (`@rohitg00`), publicada bajo la licencia MIT. El trabajo
opcional de migración, traducción y reorganización lo realiza
**Miguel Cortés** para el **Diplomado en Machine Learning** de la
**Universidad Distrital Francisco José de Caldas**.

---

## Autoría y curación

| Rol | Persona / Organización | Aporte |
|---|---|---|
| Autor original del currículo en inglés | **Rohit Ghumare** ([github.com/rohitg00](https://github.com/rohitg00)) | Diseño pedagógico, plan de estudios, código de las 503 lecciones, redacción en inglés, sitio web [aiengineeringfromscratch.com](https://aiengineeringfromscratch.com) |
| Creador de *Agent Memory* | Rohit Ghumare | Repositorio [#1 en memoria persistente para agentes](https://github.com/rohitg00/agentmemory) citado en la cabecera del repo original |
| Adaptación, traducción y migración al español | **Miguel Cortés** | Reorganización del plan del diplomado como *fases* dentro del esquema de 20, traducción de los README, plantillas y documentación, mantenimiento de la nomenclatura para la Universidad Distrital |
| Institución académica | **Universidad Distrital Francisco José de Caldas** — Diplomado en Machine Learning | Espacio institucional, estudiantes, profesores y validación pedagógica del material en español |

---

## Proyecto original (idioma inglés)

- **Nombre:** AI Engineering from Scratch
- **Repositorio:** <https://github.com/rohitg00/ai-engineering-from-scratch>
- **Sitio web:** <https://aiengineeringfromscratch.com>
- **Autor:** Rohit Ghumare
- **Licencia:** MIT
- **Tamaño:** 503 lecciones · 20 fases · ~320 horas · Python, TypeScript, Rust y Julia
- **Estadísticas (junio de 2026):** 150.639 lectores · 241.669 vistas en 30 días

### Cómo se reutiliza el material

1. **Estructura.** Las 20 fases y la organización `fase-NN/<lección>/{docs,code,outputs}`
   siguen la propuesta del repo original. Se renombró el español *Clase → Fase*
   para mantener la nomenclatura original.
2. **Plan de estudios.** Los títulos de fase, los conteos de lecciones y el orden
   pedagógico (matemáticas primero, agentes al final) provienen del README del
   repo original.
3. **Plantillas.** `PLANTILLA_LECCION.md` y `ESTRUCTURA.md` se inspiran en
   `LESSON_TEMPLATE.md` y `CONTRIBUTING.md` del repo original, adaptados a
   español.
4. **Código.** El código de cada lección del repo original **no se copia**.
   Las fases que cubrían temas ya presentes en el plan original del
   Diplomado de la Universidad Distrital usan su
   propio material (notebooks y scripts `.py`). Las fases nuevas (0, 1, 5, 6,
   7, 8, 10-19) se dejan como *esqueleto* listo para que estudiantes y
   profesores las llenen — sin duplicar el trabajo previo de Rohit.

### Lo que este repositorio **no** hace

- No clona ni redistribuye los archivos `docs/en.md`, `code/main.py`, etc.
  del repo original.
- No incluye el sitio estático (`site/`) ni los flujos de GitHub Actions
  del repo original, ya que el AGENTS.md del upstream prohíbe explícitamente
  citar repos de currículos externos en docs, código o mensajes de commit.
- No reclama autoría del plan de estudios; el crédito a Rohit Ghumare
  permanece en este archivo, en el README principal y en la cabecera de
  cada fase.

---

## Licencia

- **Este repositorio:** MIT — ver [LICENSE](./LICENSE).
- **Repositorio original (AI Engineering from Scratch):** MIT —
  Copyright (c) Rohit Ghumare.
- **Material propio de la Universidad Distrital:** MIT, mismo archivo
  [LICENSE](./LICENSE) cubre la adaptación.

El texto del MIT exige conservar el aviso de copyright y de permiso en
todas las copias. Por eso este archivo existe: para mantener el crédito al
autor original y dejar claro qué parte es de quién.

---

## Cómo citar

Si utilizas este material en una publicación, tesis o proyecto, cita ambos
trabajos:

```bibtex
@misc{ghumare2026ai_engineering_from_scratch,
  author       = {Rohit Ghumare},
  title        = {AI Engineering from Scratch},
  year         = {2026},
  howpublished = {\url{https://github.com/rohitg00/ai-engineering-from-scratch}},
  note         = {MIT License}
}

@misc{cortes2026ia_repository,
  author       = {Miguel Cortés},
  title        = {ia\_repository — Diplomado en Machine Learning,
                  Universidad Distrital Francisco José de Caldas},
  year         = {2026},
  howpublished = {\url{https://github.com/miguelcort/ia_repository}},
  note         = {Adaptación al español de AI Engineering from Scratch.
                  MIT License}
}
```

---

## Contacto

- **Repositorio original:** <https://github.com/rohitg00/ai-engineering-from-scratch/issues>
- **Adaptación al español (Universidad Distrital):** issues y PRs en este
  repositorio.
- **Sitio web del autor original:** <https://aiengineeringfromscratch.com>

¡Gracias a Rohit Ghumare por hacer accesible un plan de estudios completo
de ingeniería de IA bajo licencia MIT! 🙌
