#!/usr/bin/env python3
"""
Genera el esqueleto de una lección nueva.

Crea la estructura de carpetas, `docs/es.md` con frontmatter
rellenado, `code/main.<lang>` con cabecera, `code/tests/test_main.<lang>`
con 1 test placeholder, `quiz.json` con 6 placeholders, y
`code/README.md` con instrucciones de ejecución.

Uso:
    python3 scripts/scaffold_leccion.py \
        --fase 00 --slug entorno-desarrollo --titulo "Entorno de desarrollo" \
        --tipo Construir --lenguajes python --tiempo 45

    # Para una lección de bash/docker
    python3 scripts/scaffold_leccion.py \
        --fase 00 --slug entornos-python --titulo "Entornos virtuales de Python" \
        --tipo Construir --lenguajes bash --tiempo 30
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
FASES = RAIZ / "fases"

NOTA_ATRIBUCION = """
---

> 📚 **Adaptación al español** de la lección
> "[{titulo_original}]" del currículo
> [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch)
> (Rohit Ghumare, MIT). Implementación y documentación reescritas
> desde cero. Ver [CREDITS.md](../../../CREDITS.md).
"""


def cargar_titulo_original(idioma_original: str = "inglés") -> str:
    """Pide al usuario el título original (no se guarda en ningún archivo
    automático, se inyecta en la nota de atribución del docs/es.md)."""
    return input(f"Título original en {idioma_original} (para la nota de atribución): ").strip()


def ruta_fase(fase: str) -> Path:
    if not re.fullmatch(r"\d{2}", fase):
        sys.exit(f"--fase debe ser dos dígitos (00-19); recibí: {fase!r}")
    candidatas = sorted(FASES.glob(f"{fase}-*"))
    if not candidatas:
        sys.exit(f"No existe carpeta de fase que empiece con {fase}-")
    if len(candidatas) > 1:
        sys.exit(f"Hay varias fases con prefijo {fase}-: {[c.name for c in candidatas]}")
    return candidatas[0]


def ruta_leccion(fase_dir: Path, slug: str) -> Path:
    if not re.fullmatch(r"\d{2}-[a-z0-9-]+", slug):
        sys.exit(
            f"--slug debe tener formato NN-slug-kebab-case; recibí: {slug!r}"
        )
    lec = fase_dir / slug
    if lec.exists():
        sys.exit(f"Ya existe la lección: {lec}")
    return lec


def lenguaje_a_extension(lenguajes: list[str]) -> tuple[str, str]:
    """Devuelve (extension_archivo, runner_test) según los lenguajes."""
    lenguajes = {l.lower() for l in lenguajes}
    if "python" in lenguajes or "py" in lenguajes:
        return "py", "python3 -m unittest discover -s tests -v"
    if "bash" in lenguajes or "sh" in lenguajes or "shell" in lenguajes:
        return "sh", "# ejecutar manualmente: bash code/main.sh"
    if "docker" in lenguajes:
        return "Dockerfile", "docker build -t test ."
    if "rust" in lenguajes or "rs" in lenguajes:
        return "rs", "rustc --test --edition 2021 code/main.rs"
    if "julia" in lenguajes or "jl" in lenguajes:
        return "jl", "julia --project=. -e 'using Pkg; Pkg.test()'"
    if "typescript" in lenguajes or "ts" in lenguajes:
        return "ts", "npx tsx --test"
    return "py", "python3 -m unittest discover -s tests -v"


def contenido_docs_es(
    titulo: str,
    tipo: str,
    lenguajes: list[str],
    tiempo: int,
    titulo_original: str,
) -> str:
    return f"""# {titulo}

> <Lema de una línea: la idea central en una frase>

**Tipo:** {tipo}
**Lenguajes:** {", ".join(lenguajes)}
**Prerrequisitos:** Ninguno
**Tiempo estimado:** ~{tiempo} minutos

## Objetivos de aprendizaje

- <Verbo en infinitivo> + <objeto> + <contexto>
- <Verbo en infinitivo> + <objeto> + <contexto>
- <Verbo en infinitivo> + <objeto> + <contexto>
- 4-6 viñetas en total.

## El problema

<Describe el dolor concreto que esta lección resuelve.>

## El concepto

<Intuición y matemática mínima, si aplica. Diagramas con Mermaid o SVG.>

## Constrúyelo

<Implementación desde cero, sin frameworks.>

## Úsalo

<La misma operación con la librería o herramienta estándar.>

## Ejercicios

1. Ejercicio guiado.
2. Ejercicio con pista.
3. Ejercicio desafío (sin pistas).

## Lecturas recomendadas

- <Paper, RFC o documentación oficial>
{NOTA_ATRIBUCION.format(titulo_original=titulo_original)}
"""


def contenido_main_py(slug: str, titulo: str) -> str:
    return f'''"""
Lección: {slug}
Fase: 00
Prerrequisitos: Ninguno
Fuentes: <añadir URLs a papers, RFCs o docs oficiales>
"""
from __future__ import annotations

import sys


def main() -> int:
    """Demo auto-terminal. Imprime un resumen del entorno."""
    print("=== {titulo} ===")
    print(f"Python {{sys.version.split()[0]}}")
    print(f"Plataforma: {{sys.platform}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def contenido_main_sh(titulo: str) -> str:
    return f"""#!/usr/bin/env bash
# Lección: <slug>
# Fase: 00
# Prerrequisitos: Ninguno
# Fuentes: <añadir URLs a docs oficiales>

set -euo pipefail

echo "=== {titulo} ==="
echo "Bash: ${{BASH_VERSION}}"
echo "Sistema: $(uname -srm)"
"""


def contenido_main_dockerfile(titulo: str) -> str:
    return f"""# Lección: <slug>
# Fase: 00
# Prerrequisitos: Ninguno
# Fuentes: <añadir URLs a docs oficiales>
FROM python:3.12-slim

WORKDIR /app
COPY . /app

CMD ["python", "main.py"]
"""


def contenido_test_py() -> str:
    return '''"""Pruebas para la lección. Mínimo 5."""
import unittest


class TestLeccion(unittest.TestCase):
    def test_placeholder(self):
        """Reemplazar con la primera prueba real."""
        self.assertTrue(True)

    def test_segunda(self):
        self.assertEqual(1 + 1, 2)

    def test_tercera(self):
        self.assertIn("a", "abc")

    def test_cuarta(self):
        items = [1, 2, 3]
        self.assertEqual(len(items), 3)

    def test_quinta(self):
        d = {"a": 1}
        self.assertEqual(d["a"], 1)


if __name__ == "__main__":
    unittest.main()
'''


def contenido_test_sh() -> str:
    return """#!/usr/bin/env bash
# Pruebas para la lección. Mínimo 5.
set -euo pipefail

fail() { echo "FAIL: $*"; exit 1; }
pass() { echo "  ok: $*"; }

test_placeholder() { pass "placeholder"; }
test_dos() { [ "$((1+1))" = "2" ] && pass "1+1=2"; }
test_tres() { echo "abc" | grep -q "a" && pass "contiene a"; }
test_cuatro() { [ "$(printf '%s\\n' a b c | wc -l)" = "3" ] && pass "3 lineas"; }
test_cinco() { [ -n "x" ] && pass "x no es vacio"; }

test_placeholder
test_dos
test_tres
test_cuatro
test_cinco

echo "OK: 5 pruebas pasadas"
"""


def contenido_quiz(slug: str, titulo: str) -> str:
    return f"""{{
  "lesson": "{slug}",
  "title": "{titulo}",
  "questions": [
    {{
      "stage": "pre",
      "question": "<Pregunta previa>",
      "options": ["<opción a>", "<opción b>", "<opción c>", "<opción d>"],
      "correct": 0,
      "explanation": "<Explicación>"
    }},
    {{
      "stage": "check",
      "question": "<Pregunta de comprobación 1>",
      "options": ["<opción a>", "<opción b>", "<opción c>", "<opción d>"],
      "correct": 1,
      "explanation": "<Explicación>"
    }},
    {{
      "stage": "check",
      "question": "<Pregunta de comprobación 2>",
      "options": ["<opción a>", "<opción b>", "<opción c>", "<opción d>"],
      "correct": 2,
      "explanation": "<Explicación>"
    }},
    {{
      "stage": "check",
      "question": "<Pregunta de comprobación 3>",
      "options": ["<opción a>", "<opción b>", "<opción c>", "<opción d>"],
      "correct": 1,
      "explanation": "<Explicación>"
    }},
    {{
      "stage": "post",
      "question": "<Pregunta posterior 1>",
      "options": ["<opción a>", "<opción b>", "<opción c>", "<opción d>"],
      "correct": 3,
      "explanation": "<Explicación>"
    }},
    {{
      "stage": "post",
      "question": "<Pregunta posterior 2>",
      "options": ["<opción a>", "<opción b>", "<opción c>", "<opción d>"],
      "correct": 0,
      "explanation": "<Explicación>"
    }}
  ]
}}
"""


def contenido_code_readme(titulo: str, runner: str) -> str:
    return f"""# {titulo}

Ejecuta el demo:

```bash
{runner}
```
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fase", required=True, help="Número de fase (00-19).")
    parser.add_argument("--slug", required=True, help="Slug kebab-case con prefijo NN-.")
    parser.add_argument("--titulo", required=True, help="Título de la lección en español.")
    parser.add_argument(
        "--tipo",
        default="Construir",
        choices=["Aprender", "Construir", "Referencia"],
    )
    parser.add_argument(
        "--lenguajes",
        default="python",
        help="Lista separada por comas (python, bash, docker, rust, julia, typescript).",
    )
    parser.add_argument("--tiempo", type=int, default=30, help="Tiempo en minutos.")
    parser.add_argument(
        "--titulo-original",
        default=None,
        help="Título original en inglés. Si se omite, se pregunta interactivamente.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="No pedir confirmación interactiva.",
    )
    args = parser.parse_args(argv)

    lenguajes = [l.strip() for l in args.lenguajes.split(",") if l.strip()]
    extension, runner = lenguaje_a_extension(lenguajes)
    fase_dir = ruta_fase(args.fase)
    lec_dir = ruta_leccion(fase_dir, args.slug)

    titulo_original = args.titulo_original
    if not titulo_original:
        if args.yes:
            titulo_original = args.titulo
        else:
            titulo_original = cargar_titulo_original()

    if not args.yes:
        print(f"\nVas a crear: {lec_dir.relative_to(RAIZ)}")
        print(f"  Título:    {args.titulo}")
        print(f"  Tipo:      {args.tipo}")
        print(f"  Lenguajes: {lenguajes}")
        print(f"  Tiempo:    ~{args.tiempo} min")
        print(f"  Original:  {titulo_original}")
        r = input("\n¿Continuar? [s/N] ")
        if r.strip().lower() not in {"s", "si", "sí", "y", "yes"}:
            print("Cancelado.")
            return 1

    (lec_dir / "docs").mkdir(parents=True)
    (lec_dir / "code" / "tests").mkdir(parents=True)

    (lec_dir / "docs" / "es.md").write_text(
        contenido_docs_es(args.titulo, args.tipo, lenguajes, args.tiempo, titulo_original),
        encoding="utf-8",
    )

    if extension == "py":
        (lec_dir / "code" / "main.py").write_text(
            contenido_main_py(args.slug, args.titulo), encoding="utf-8"
        )
        (lec_dir / "code" / "tests" / "test_main.py").write_text(
            contenido_test_py(), encoding="utf-8"
        )
    elif extension == "sh":
        main_path = lec_dir / "code" / "main.sh"
        main_path.write_text(contenido_main_sh(args.titulo), encoding="utf-8")
        main_path.chmod(0o755)
        test_path = lec_dir / "code" / "tests" / "test_main.sh"
        test_path.write_text(contenido_test_sh(), encoding="utf-8")
        test_path.chmod(0o755)
    elif extension == "Dockerfile":
        (lec_dir / "code" / "Dockerfile").write_text(
            contenido_main_dockerfile(args.titulo), encoding="utf-8"
        )
        (lec_dir / "code" / "tests" / "test_main.py").write_text(
            contenido_test_py(), encoding="utf-8"
        )
    else:
        (lec_dir / "code" / f"main.{extension}").write_text(
            contenido_main_py(args.slug, args.titulo), encoding="utf-8"
        )
        (lec_dir / "code" / "tests" / f"test_main.{extension}").write_text(
            contenido_test_py(), encoding="utf-8"
        )

    (lec_dir / "code" / "README.md").write_text(
        contenido_code_readme(args.titulo, runner), encoding="utf-8"
    )
    (lec_dir / "quiz.json").write_text(
        contenido_quiz(args.slug, args.titulo), encoding="utf-8"
    )

    print(f"\n✓ Esqueleto creado en {lec_dir.relative_to(RAIZ)}")
    print("  Siguiente paso: editar docs/es.md, code/main.* y los tests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
