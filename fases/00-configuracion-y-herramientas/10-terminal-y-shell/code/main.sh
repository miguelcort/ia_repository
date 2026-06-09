#!/usr/bin/env bash
# Lección: 10-terminal-y-shell
# Fase: 00
# Prerrequisitos: 01-entorno-desarrollo
# Fuentes:
# - Bash manual: https://www.gnu.org/software/bash/manual/
# - ShellCheck: https://www.shellcheck.net/

set -euo pipefail

uso() {
  cat <<EOF
Uso: $(basename "$0") <comando> [args]

Comandos:
  info               Muestra info del shell actual
  count-words <f>    Cuenta lineas y palabras de un archivo
  find-py <dir>      Encuentra archivos .py en un directorio
  count-todos <dir>  Cuenta TODOs en archivos .py
  tail-log <f>       Imprime las ultimas 10 lineas de un log
  env-python         Muestra info del Python en PATH
  diff-size <a> <b>  Compara tamano de dos archivos
  snippet <f>        Imprime lineas 1-10 de un archivo
  json-pretty <f>    Pretty-print de un JSON
  find-large <dir>   Lista archivos > 10MB en un directorio
  random-id          Genera un identificador aleatorio
  fecha              Fecha actual en formato ISO
EOF
}

info() {
  echo "Shell: ${SHELL:-desconocido}"
  echo "Bash: ${BASH_VERSION:-no es bash}"
  echo "Usuario: ${USER:-desconocido}"
  echo "PWD: ${PWD:-desconocido}"
}

count_words() {
  local archivo="${1:-}"
  if [ -z "$archivo" ] || [ ! -f "$archivo" ]; then
    echo "ERROR: archivo no valido"
    return 1
  fi
  local lineas palabras
  lineas=$(wc -l < "$archivo" | tr -d ' ')
  palabras=$(wc -w < "$archivo" | tr -d ' ')
  echo "Lineas: $lineas, Palabras: $palabras"
}

find_py() {
  local dir="${1:-.}"
  find "$dir" -name "*.py" -type f 2>/dev/null | head -20
}

count_todos() {
  local dir="${1:-.}"
  grep -rn "TODO\|FIXME" --include="*.py" "$dir" 2>/dev/null | wc -l | tr -d ' '
}

tail_log() {
  local archivo="${1:-}"
  [ -f "$archivo" ] && tail -10 "$archivo"
}

env_python() {
  command -v python3 || echo "python3 no en PATH"
  python3 --version 2>&1 || true
}

diff_size() {
  [ $# -eq 2 ] || { echo "ERROR: necesita 2 archivos"; return 1; }
  local s1 s2
  s1=$(stat -f%z "$1" 2>/dev/null || stat -c%s "$1" 2>/dev/null)
  s2=$(stat -f%z "$2" 2>/dev/null || stat -c%s "$2" 2>/dev/null)
  echo "$1: $s1 bytes, $2: $s2 bytes"
}

snippet() {
  head -10 "${1:-}"
}

json_pretty() {
  python3 -m json.tool < "${1:-/dev/stdin}"
}

find_large() {
  local dir="${1:-.}"
  find "$dir" -type f -size +10M 2>/dev/null | head -10
}

random_id() {
  cat /dev/urandom | head -c 8 | od -An -tx1 | tr -d ' \n'
  echo
}

fecha() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

case "${1:-}" in
  info) info ;;
  count-words) shift; count_words "$@" ;;
  find-py) shift; find_py "$@" ;;
  count-todos) shift; count_todos "$@" ;;
  tail-log) shift; tail_log "$@" ;;
  env-python) env_python ;;
  diff-size) shift; diff_size "$@" ;;
  snippet) shift; snippet "$@" ;;
  json-pretty) shift; json_pretty "$@" ;;
  find-large) shift; find_large "$@" ;;
  random-id) random_id ;;
  fecha) fecha ;;
  *) uso ;;
esac
