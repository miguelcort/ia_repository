#!/usr/bin/env bash
# Lección: 02-git-colaboracion
# Fase: 00
# Prerrequisitos: 01-entorno-desarrollo
# Fuentes:
# - Pro Git (libre): https://git-scm.com/book/es/v2
# - Conventional Commits: https://www.conventionalcommits.org/es/

set -euo pipefail

CONVENCION='^(feat|fix|docs|style|refactor|test|chore|perf|build|ci)\([a-z0-9-]+/[0-9]+\): .+.{1,72}$'
MAX_LONGITUD=72

uso() {
  cat <<EOF
Uso: $(basename "$0") <opcion>

  check    Verifica que el repo esta bien configurado
  sample   Verifica un mensaje de commit de ejemplo
  report   Muestra un informe del estado del repo
EOF
}

verificar_repo() {
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "ERROR: no estas dentro de un repositorio git"
    return 1
  fi
  local rama
  rama=$(git rev-parse --abbrev-ref HEAD)
  echo "OK: repo valido en rama '$rama'"
}

verificar_identidad() {
  local nombre correo
  nombre=$(git config user.name || echo "")
  correo=$(git config user.email || echo "")
  if [ -z "$nombre" ] || [ -z "$correo" ]; then
    echo "FALTA: configura tu identidad con:"
    echo "  git config user.name  'Tu Nombre'"
    echo "  git config user.email 'tu@correo'"
    return 1
  fi
  echo "OK: identidad = $nombre <$correo>"
}

verificar_mensaje() {
  local asunto="$1"
  if [ ${#asunto} -gt $MAX_LONGITUD ]; then
    echo "FALLA: asunto de ${#asunto} caracteres (>${MAX_LONGITUD})"
    return 1
  fi
  if ! [[ "$asunto" =~ $CONVENCION ]]; then
    echo "FALLA: '$asunto' no sigue feat(fase-NN/MM): <slug>"
    return 1
  fi
  echo "OK: '$asunto' cumple la convencion"
}

reporte() {
  echo "=== Reporte del repositorio ==="
  echo "Rama actual:    $(git rev-parse --abbrev-ref HEAD)"
  echo "Ultimo commit:  $(git log -1 --format='%h %s' 2>/dev/null || echo 'sin commits')"
  echo "Cambios staged: $(git diff --cached --name-only | wc -l | tr -d ' ')"
  echo "Cambios sin stage: $(git diff --name-only | wc -l | tr -d ' ')"
}

case "${1:-}" in
  check) verificar_repo && verificar_identidad ;;
  sample) verificar_mensaje "${2:-feat(fase-00/02): ejemplo-de-mensaje}" ;;
  report) verificar_repo >/dev/null && reporte ;;
  *) uso ;;
esac
