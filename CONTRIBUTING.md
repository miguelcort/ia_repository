# Guía de Contribución

¡Gracias por tu interés en contribuir al repositorio del Diplomado en
Machine Learning de la **Universidad Distrital Francisco José de
Caldas**, en su versión en español del currículo
[AI Engineering from
Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) de
Rohit Ghumare!

> Antes de contribuir, lee [CREDITS.md](./CREDITS.md) y
> [PLANTILLA_LECCION.md](./PLANTILLA_LECCION.md). Toda lección debe
> respetar la atribución al autor original.

---

## ⚙️ Reglas duras (no negociables)

1. **Un commit por lección.** Nunca agrupar varias lecciones en un
   solo commit. Un PR de 5 lecciones = 5 commits.
2. **Asunto de commit `<72` caracteres**, formato:
   `feat(fase-NN/MM): <slug>`. El cuerpo explica el *por qué*, no el
   *qué*.
3. **Documentación en español** (`docs/es.md`). El código va en
   inglés por convención de la comunidad Python.
4. **Diagramas en Mermaid o SVG.** Nunca usar caracteres Unicode para
   "dibujar" diagramas.
5. **Todo bloque de código lleva etiqueta de lenguaje:**
   `python`, `typescript`, `rust`, `julia`, `bash`, `json`, `mermaid`,
   `yaml`, `text`, `console`.
6. **No duplicar código** de otros repos de currículos. Citar papers,
   RFCs y documentación oficial cuando sea la fuente canónica.
7. **No committear archivos generados.** El `.gitignore` raíz cubre
   `__pycache__/`, datasets grandes, checkpoints, etc.
8. **Licencia MIT** se mantiene en todo el repositorio.

---

## 📚 Cómo añadir una nueva lección

### 1. Crear la carpeta

```bash
# Desde la raíz del repositorio
FASE=02
SLUG=regresion-lineal-desde-cero
mkdir -p "fases/${FASE}-fundamentos-ml/${SLUG}"/{docs,code/tests,notebooks,ejercicios,referencias,outputs}
```

### 2. Escribir `docs/es.md`

Usa la [PLANTILLA_LECCION.md](./PLANTILLA_LECCION.md). Frontmatter
obligatorio:

```markdown
# <Título>

> <Lema de una línea>

**Tipo:** <Aprender | Construir | Referencia>
**Lenguajes:** <lista separada por comas>
**Prerrequisitos:** <lecciones previas o "Ninguno">
**Tiempo estimado:** ~<minutos> minutos

## Objetivos de aprendizaje
- <Verbo en infinitivo> + <objeto> + <contexto>
- 4 a 6 viñetas.

## El problema
## El concepto
## Constrúyelo
## Úsalo
## Despliégalo
## Ejercicios
## Lecturas recomendadas
```

### 3. Implementar `code/main.<lang>`

```python
"""
Lección: regresion-lineal-desde-cero
Fase: 02
Prerrequisitos: 01-algebra-lineal
Fuentes: NumPy docs, ESL cap. 3
"""
# implementación...
```

Reglas:

- Cabecera de 4-6 líneas con la ruta a `docs/es.md` y las fuentes.
- El demo `__main__` debe **auto-terminar** (sin `input()`, sin bucles
  infinitos).
- Sin comentarios triviales; la documentación explica el *por qué*.

### 4. Escribir los tests

Mínimo **5 pruebas unitarias** en `code/tests/test_main.<lang>`. Deben
poder correr con `python3 -m unittest discover -s code/tests -v`.

### 5. (Opcional) Artefacto reutilizable

Si la lección produce un *prompt*, *skill*, *agente* o *servidor MCP*,
añádelo en `outputs/<tipo>-<slug>.md` con frontmatter YAML.

### 6. Actualizar el `README.md` de la fase

Edita `fases/NN-nombre-fase/README.md` y marca la lección como ✅.

### 7. Validar localmente

```bash
# Auditoría general
python3 scripts/auditar_lecciones.py

# Lección específica
cd fases/02-fundamentos-ml/02-regresion-lineal-desde-cero/code
python3 main.py
python3 -m unittest discover tests -v
```

### 8. Commit y PR

```bash
git add fases/02-fundamentos-ml/02-regresion-lineal-desde-cero \
        fases/02-fundamentos-ml/README.md

git commit -m "feat(fase-02/02): regresion-lineal-desde-cero"

git push -u origin feature/fase-02-regresion-lineal
gh pr create --title "feat(fase-02/02): regresion-lineal-desde-cero" \
            --body "Implementación desde cero de regresión lineal en NumPy."
```

---

## 🐛 Cómo reportar errores

1. Verifica que no esté ya reportado en los Issues.
2. Crea un nuevo Issue con título claro y pasos para reproducir.
3. Si el error está en una lección específica, indica
   `fase-NN/MM-slug` y el archivo `docs/es.md` afectado.

---

## 💡 Cómo sugerir mejoras

- Abre un Issue con la etiqueta `enhancement`.
- Explica el problema que resuelve y, si es posible, incluye un
  ejemplo o borrador.

---

## 🌐 Cómo añadir una traducción

Las traducciones del español a otros idiomas siguen el mismo
esquema que el repo original (`docs/<codigo-iso>.md`):

```text
docs/
├── es.md      # Español (siempre presente)
├── en.md      # Inglés (si existe)
├── pt.md      # Portugués (nuevo)
└── ...
```

La traducción **debe mantener la misma estructura y ejemplos** que la
versión en español.

---

## 📏 Estándares de código

### Python

- PEP 8 + Black (longitud de línea 88).
- Type hints en funciones públicas.
- Docstrings sólo cuando aporten información que el nombre no da.
- Sin comentarios triviales; la documentación está en `docs/es.md`.

### Jupyter notebooks

- Primera celda con título y resumen.
- Markdown para explicar *por qué*, no *qué*.
- Limpia los outputs antes de commitear.
- Orden lógico: imports → carga de datos → experimento → análisis.

### Documentación

- Español como idioma principal.
- Tono directo, sin marketing ni filler.
- Sin emojis decorativos en encabezados (excepto el badge del README
  principal).
- Diagramas en Mermaid, no en caracteres Unicode.

---

## 📂 Estructura de commits

```text
tipo(alcance): asunto breve

Descripción más detallada del por qué.
```

Tipos:

- `feat` — Nueva lección o funcionalidad.
- `fix` — Corrección de error.
- `docs` — Cambios de documentación.
- `refactor` — Refactorización sin cambio de comportamiento.
- `test` — Añadir o mejorar tests.
- `chore` — Mantenimiento general (CI, dependencias, etc.).

Ejemplos:

```text
feat(fase-02/02): regresion-lineal-desde-cero

Implementación de regresión lineal en NumPy con 6 pruebas
unitarias y notas sobre la elección del learning rate.
```

```text
fix(fase-04/06): yolo-loss-shape-mismatch

El tensor de targets tenía dimensión incorrecta al pasar
por el loss; corregido con un reshape explícito.
```

---

## 🔍 Revisión de código

Todo PR será revisado por:

- Correcto funcionamiento del código (`python3 main.py` exit 0).
- 5+ tests pasando.
- Documentación completa en `docs/es.md`.
- Sin conflictos con la rama principal.
- Cumplimiento de las reglas duras arriba.

---

## 📜 Licencia y atribución

Al contribuir, aceptas que tu código se licencie bajo MIT, la misma
licencia del proyecto y del currículo original. Ver
[LICENSE](./LICENSE) y [CREDITS.md](./CREDITS.md).

---

## 💬 Código de Conducta

Ver [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md). Sé amable, sé útil,
sé constructivo. Este es un espacio de aprendizaje.

---

## 🎓 Reconocimientos

Los contribuidores serán listados en la sección *Agradecimientos* del
[README.md](./README.md) y referenciados en la cabecera de las
lecciones en las que participen.

---

¡Gracias por ayudar a construir un currículo de IA completo en
español! 🙌
