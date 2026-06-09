#!/usr/bin/env bash
# Pruebas para 10-terminal-y-shell

set -uo pipefail

fail() { echo "FAIL: $*" >&2; exit 1; }
pass() { echo "  ok: $*"; }

RAIZ_REPO="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$RAIZ_REPO"

SCRIPT="$RAIZ_REPO/fases/00-configuracion-y-herramientas/10-terminal-y-shell/code/main.sh"

[ -x "$SCRIPT" ] || fail "main.sh no es ejecutable"
pass "main.sh existe y es ejecutable"

# Sin argumentos muestra uso
uso=$(bash "$SCRIPT" 2>&1)
[[ "$uso" == *"Uso:"* ]] || fail "sin argumentos no imprime uso"
pass "sin argumentos muestra ayuda"

# info funciona
salida=$(bash "$SCRIPT" info 2>&1)
[[ "$salida" == *"Shell:"* ]] || fail "info no imprime shell"
[[ "$salida" == *"PWD:"* ]] || fail "info no imprime PWD"
pass "info reporta shell y PWD"

# env-python
salida=$(bash "$SCRIPT" env-python 2>&1)
[[ "$salida" == *"Python"* ]] || fail "env-python no detecta Python"
pass "env-python detecta Python"

# count-words con archivo valido
TMP=$(mktemp)
printf 'hola mundo\nsegunda linea\n' > "$TMP"
salida=$(bash "$SCRIPT" count-words "$TMP" 2>&1)
[[ "$salida" == *"Lineas: 2"* ]] || fail "count-words: $salida"
[[ "$salida" == *"Palabras: 4"* ]] || fail "count-words: palabras"
rm -f "$TMP"
pass "count-words cuenta lineas y palabras"

# count-words con archivo invalido
salida=$(bash "$SCRIPT" count-words /no/existe.txt 2>&1 || true)
[[ "$salida" == *"ERROR"* ]] || fail "count-words no falla con archivo invalido"
pass "count-words falla con archivo invalido"

# random-id genera 16 caracteres hex
salida=$(bash "$SCRIPT" random-id 2>&1)
[[ ${#salida} -eq 16 ]] || fail "random-id no genera 16 chars: '$salida' (${#salida})"
[[ "$salida" =~ ^[0-9a-f]+$ ]] || fail "random-id no es hex: $salida"
pass "random-id genera 16 caracteres hex"

# fecha formato ISO
salida=$(bash "$SCRIPT" fecha 2>&1)
[[ "$salida" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T ]] || fail "fecha no es ISO: $salida"
pass "fecha devuelve formato ISO 8601"

# snippet imprime 10 lineas
TMP=$(mktemp)
seq 1 30 > "$TMP"
salida=$(bash "$SCRIPT" snippet "$TMP" 2>&1)
[[ $(echo "$salida" | wc -l) -eq 10 ]] || fail "snippet no imprime 10 lineas"
rm -f "$TMP"
pass "snippet imprime 10 lineas"

# diff-size compara dos archivos
TMP1=$(mktemp); TMP2=$(mktemp)
echo "abc" > "$TMP1"
echo "abcdef" > "$TMP2"
salida=$(bash "$SCRIPT" diff-size "$TMP1" "$TMP2" 2>&1)
[[ "$salida" == *"bytes"* ]] || fail "diff-size no imprime bytes: $salida"
rm -f "$TMP1" "$TMP2"
pass "diff-size compara archivos"

# json-pretty
TMP=$(mktemp)
echo '{"a":1,"b":2}' > "$TMP"
salida=$(bash "$SCRIPT" json-pretty "$TMP" 2>&1)
[[ "$salida" == *"\"a\""* ]] || fail "json-pretty no funciona: $salida"
rm -f "$TMP"
pass "json-pretty formatea JSON"

# find-py encuentra archivos .py
salida=$(bash "$SCRIPT" find-py "$RAIZ_REPO/scripts" 2>&1)
[[ "$salida" == *".py"* ]] || fail "find-py no encuentra .py: $salida"
pass "find-py encuentra archivos .py"

# comando desconocido
salida=$(bash "$SCRIPT" comando_inexistente_xyz 2>&1 || true)
[[ "$salida" == *"Uso:"* ]] || fail "comando desconocido no imprime uso"
pass "comando desconocido imprime uso"

echo "OK: 12 pruebas pasadas"
