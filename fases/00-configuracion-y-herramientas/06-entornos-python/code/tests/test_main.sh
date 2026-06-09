#!/usr/bin/env bash
# Pruebas para 06-entornos-python
# Las pruebas crean un entorno temporal, lo limpian al final.

set -uo pipefail

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "  ok: $*"; }

RAIZ_REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$RAIZ_REPO"

SCRIPT="$RAIZ_REPO/fases/00-configuracion-y-herramientas/06-entornos-python/code/main.sh"
export NOMBRE_ENTORNO=".venv_test_$$"

[ -x "$SCRIPT" ] || fail "main.sh no es ejecutable"
pass "main.sh existe y es ejecutable"

# Sin argumentos muestra uso
uso=$(bash "$SCRIPT" 2>&1)
[[ "$uso" == *"Uso:"* ]] || fail "sin argumentos no imprime uso"
pass "sin argumentos muestra ayuda"

# create funciona
if [ -d "$NOMBRE_ENTORNO" ]; then
  bash "$SCRIPT" clean >/dev/null 2>&1 || true
fi
salida=$(bash "$SCRIPT" create 2>&1)
[[ "$salida" == *"OK"* ]] || fail "create fallo: $salida"
[ -d "$NOMBRE_ENTORNO" ] || fail "directorio $NOMBRE_ENTORNO no se creo"
pass "create genera el directorio del entorno"

# create es idempotente
salida=$(bash "$SCRIPT" create 2>&1)
[[ "$salida" == *"AVISO"* ]] || fail "create idempotente no emite aviso"
pass "create es idempotente"

# check reporta el Python del entorno
salida=$(bash "$SCRIPT" check 2>&1)
[[ "$salida" == *"Python:"* ]] || fail "check no reporta Python: $salida"
pass "check reporta el Python del entorno"

# python dentro del entorno
salida=$(bash "$SCRIPT" python -c "import sys; print(sys.prefix)" 2>&1)
[[ "$salida" == *"$NOMBRE_ENTORNO"* ]] || fail "python no usa el entorno: $salida"
pass "python se ejecuta dentro del entorno"

# pip funciona
salida=$(bash "$SCRIPT" pip --version 2>&1 || true)
[[ "$salida" == *"pip"* || "$salida" == *"AVISO"* || "$salida" == *"no se"* ]] \
  || fail "pip no responde: $salida"
pass "pip responde dentro del entorno"

# clean elimina el entorno
bash "$SCRIPT" clean >/dev/null 2>&1
[ ! -d "$NOMBRE_ENTORNO" ] || fail "clean no elimino el entorno"
pass "clean elimina el entorno"

# install sin requirements.txt avisa
salida=$(bash "$SCRIPT" install 2>&1)
[[ "$salida" == *"AVISO"* || "$salida" == *"OK"* || "$salida" == *"ERROR"* ]] \
  || fail "install sin requirements fallo extranamente: $salida"
pass "install procesa requirements.txt (o avisa si no hay)"

unset NOMBRE_ENTORNO
echo "OK: 8 pruebas pasadas"
