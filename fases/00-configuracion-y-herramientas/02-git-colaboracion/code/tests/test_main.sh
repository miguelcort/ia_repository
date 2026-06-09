#!/usr/bin/env bash
# Pruebas para la lección 02-git-colaboracion.
# Las pruebas se ejecutan desde la raíz del repo (donde está .git).

set -uo pipefail

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "  ok: $*"; }

RAIZ_REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$RAIZ_REPO"

SCRIPT="$RAIZ_REPO/fases/00-configuracion-y-herramientas/02-git-colaboracion/code/main.sh"

[ -x "$SCRIPT" ] || fail "main.sh no es ejecutable"
pass "main.sh existe y es ejecutable"

# check: debe ejecutarse sin error (estamos en el repo ia_repository)
bash "$SCRIPT" check >/dev/null
pass "modo check OK dentro de un repo git"

# report: debe ejecutarse sin error
bash "$SCRIPT" report >/dev/null
pass "modo report OK"

# sample: mensaje válido debe pasar
resultado=$(bash "$SCRIPT" sample 2>&1)
[[ "$resultado" == *"OK"* ]] || fail "sample no reconoció mensaje válido: $resultado"
pass "sample acepta feat(fase-NN/MM): slug"

# sample con mensaje inválido debe fallar
resultado=$(bash "$SCRIPT" sample "mensaje libre" 2>&1 || true)
[[ "$resultado" == *"FALLA"* ]] || fail "sample aceptó mensaje inválido"
pass "sample rechaza mensaje sin prefijo de tipo"

# sample con asunto de más de 72 caracteres
largo=$(printf 'a%.0s' {1..80})
resultado=$(bash "$SCRIPT" sample "$largo" 2>&1 || true)
[[ "$resultado" == *"FALLA"* ]] || fail "sample aceptó asunto >72"
pass "sample rechaza asunto de más de 72 caracteres"

# sample con tipo no convencional
resultado=$(bash "$SCRIPT" sample "inventado(fase-00/02): cosa" 2>&1 || true)
[[ "$resultado" == *"FALLA"* ]] || fail "sample aceptó tipo no convencional"
pass "sample rechaza tipo fuera del set permitido"

# Modo sin argumentos imprime uso
uso=$(bash "$SCRIPT" 2>&1)
[[ "$uso" == *"Uso:"* ]] || fail "sin argumentos no imprime uso"
pass "sin argumentos muestra ayuda"

echo "OK: 8 pruebas pasadas"
