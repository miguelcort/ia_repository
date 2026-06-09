#!/usr/bin/env bash
# Lección: 06-entornos-python
# Fase: 00
# Prerrequisitos: 01-entorno-desarrollo
# Fuentes:
# - venv: https://docs.python.org/3/library/venv.html
# - pip: https://pip.pypa.io/en/stable/

set -euo pipefail

NOMBRE_ENTORNO="${NOMBRE_ENTORNO:-.venv}"
PYTHON="${PYTHON:-python3}"
COMANDO="${1:-}"

uso() {
  cat <<EOF
Uso: $(basename "$0") <comando> [argumentos]

Comandos:
  create           Crea un entorno virtual en $NOMBRE_ENTORNO
  install          Crea el entorno e instala requirements.txt
  check            Verifica que el entorno existe y reporta su estado
  python <args>    Ejecuta Python dentro del entorno
  pip <args>       Ejecuta pip dentro del entorno
  clean            Elimina el entorno

Variable de entorno NOMBRE_ENTORNO para cambiar el nombre (default: .venv).
EOF
}

verificar_python() {
  if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON no esta en PATH"
    return 1
  fi
  local version
  version="$($PYTHON --version 2>&1)"
  echo "OK: $version"
}

crear_entorno() {
  if [ -d "$NOMBRE_ENTORNO" ]; then
    echo "AVISO: $NOMBRE_ENTORNO ya existe, no se recrea"
    return 0
  fi
  verificar_python
  "$PYTHON" -m venv "$NOMBRE_ENTORNO"
  echo "OK: entorno creado en $NOMBRE_ENTORNO"
}

instalar_dependencias() {
  crear_entorno
  if [ ! -f "requirements.txt" ]; then
    echo "AVISO: no hay requirements.txt, nada que instalar"
    return 0
  fi
  "$NOMBRE_ENTORNO/bin/pip" install --upgrade pip >/dev/null
  "$NOMBRE_ENTORNO/bin/pip" install -r requirements.txt
  echo "OK: dependencias instaladas"
}

reporte() {
  if [ ! -d "$NOMBRE_ENTORNO" ]; then
    echo "FALTA: $NOMBRE_ENTORNO no existe"
    return 1
  fi
  local python_venv
  python_venv="$("$NOMBRE_ENTORNO/bin/python" --version 2>&1)"
  local paquetes
  paquetes="$("$NOMBRE_ENTORNO/bin/pip" list 2>/dev/null | wc -l | tr -d ' ')"
  echo "Entorno: $NOMBRE_ENTORNO"
  echo "Python:  $python_venv"
  echo "Paquetes: $paquetes"
}

ejecutar_python() {
  if [ ! -x "$NOMBRE_ENTORNO/bin/python" ]; then
    echo "ERROR: entorno no existe. Ejecuta '$0 create' primero"
    return 1
  fi
  "$NOMBRE_ENTORNO/bin/python" "$@"
}

ejecutar_pip() {
  if [ ! -x "$NOMBRE_ENTORNO/bin/pip" ]; then
    echo "ERROR: entorno no existe"
    return 1
  fi
  "$NOMBRE_ENTORNO/bin/pip" "$@"
}

limpiar() {
  if [ -d "$NOMBRE_ENTORNO" ]; then
    rm -rf "$NOMBRE_ENTORNO"
    echo "OK: $NOMBRE_ENTORNO eliminado"
  else
    echo "AVISO: nada que limpiar"
  fi
}

case "$COMANDO" in
  create)  crear_entorno ;;
  install) instalar_dependencias ;;
  check)   reporte ;;
  python)  shift; ejecutar_python "$@" ;;
  pip)     shift; ejecutar_pip "$@" ;;
  clean)   limpiar ;;
  *)       uso ;;
esac
