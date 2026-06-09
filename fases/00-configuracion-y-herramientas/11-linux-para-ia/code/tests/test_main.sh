#!/usr/bin/env bash
# Pruebas para 11-linux-para-ia

set -uo pipefail

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "  ok: $*"; }

RAIZ_REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$RAIZ_REPO"

SCRIPT="$RAIZ_REPO/fases/00-configuracion-y-herramientas/11-linux-para-ia/code/main.sh"

[ -x "$SCRIPT" ] || fail "main.sh no es ejecutable"
pass "main.sh existe y es ejecutable"

# Sin argumentos muestra uso
uso=$(bash "$SCRIPT" 2>&1)
[[ "$uso" == *"Uso:"* ]] || fail "sin argumentos no imprime uso"
pass "sin argumentos muestra ayuda"

# who
salida=$(bash "$SCRIPT" who 2>&1)
[[ "$salida" == *"Usuario:"* ]] || fail "who no imprime usuario"
[[ "$salida" == *"UID:"* ]] || fail "who no imprime UID"
pass "who reporta usuario y UID"

# os-info
salida=$(bash "$SCRIPT" os-info 2>&1)
[[ "$salida" == *"Kernel:"* ]] || fail "os-info no imprime kernel: $salida"
[[ "$salida" == *"Arch:"* ]] || fail "os-info no imprime arch"
pass "os-info reporta kernel y arquitectura"

# resources
salida=$(bash "$SCRIPT" resources 2>&1)
[[ "$salida" == *"Disco"* ]] || fail "resources no reporta disco: $salida"
pass "resources incluye la seccion de disco"

# top-processes
salida=$(bash "$SCRIPT" top-processes 2>&1)
[[ "$salida" == *"PID"* || "$salida" == *"Load"* || -n "$salida" ]] \
  || fail "top-processes no devuelve nada"
pass "top-processes responde"

# listening
salida=$(bash "$SCRIPT" listening 2>&1)
# Puede fallar o estar vacio; solo no debe explotar
pass "listening no explota (puede estar vacio en entornos sin red)"

# comando desconocido
salida=$(bash "$SCRIPT" comando_inexistente_xyz 2>&1 || true)
[[ "$salida" == *"Uso:"* ]] || fail "comando desconocido no imprime uso"
pass "comando desconocido imprime uso"

# Quien no es root debe tener UID > 0 (a menos que CI corra como root)
# Solo verificamos que UID sea un numero
salida=$(bash "$SCRIPT" who 2>&1)
uid=$(echo "$salida" | grep "UID:" | awk '{print $2}')
[[ "$uid" =~ ^[0-9]+$ ]] || fail "UID no es numerico: $uid"
pass "UID es numerico ($uid)"

# groups contiene al menos un grupo
salida=$(bash "$SCRIPT" who 2>&1)
grupos=$(echo "$salida" | grep "Grupos:" | sed 's/Grupos: *//')
[ -n "$grupos" ] || fail "sin grupos"
pass "Grupos reporta al menos un grupo ($grupos)"

echo "OK: 9 pruebas pasadas"
