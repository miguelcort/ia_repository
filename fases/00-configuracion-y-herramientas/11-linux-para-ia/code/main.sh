#!/usr/bin/env bash
# Lección: 11-linux-para-ia
# Fase: 00
# Prerrequisitos: 01-entorno-desarrollo, 10-terminal-y-shell
# Fuentes:
# - systemd: https://www.freedesktop.org/software/systemd/man/systemd.html
# - ss(8): https://man7.org/linux/man-pages/man8/ss.8.html

set -euo pipefail

uso() {
  cat <<EOF
Uso: $(basename "$0") <comando>

Comandos:
  who             Quien soy y a que grupos pertenezco
  resources       CPU, RAM, disco disponible
  top-processes   Top 5 procesos por uso de CPU
  listening       Puertos TCP abiertos
  os-info         Info del sistema operativo
EOF
}

quien() {
  echo "Usuario: $(id -un)"
  echo "UID:     $(id -u)"
  echo "Grupos:  $(id -Gn)"
  echo "Home:    $HOME"
}

recursos() {
  echo "== CPU =="
  if command -v nproc >/dev/null 2>&1; then
    nproc
  else
    sysctl -n hw.ncpu 2>/dev/null || echo "?"
  fi
  echo ""
  echo "== Memoria =="
  if [ -f /proc/meminfo ]; then
    grep -E "MemTotal|MemAvailable|MemFree" /proc/meminfo | head -3
  else
    vm_stat | head -5
  fi
  echo ""
  echo "== Disco =="
  df -h / 2>/dev/null | head -3
}

top_procesos() {
  if [ -f /proc/loadavg ]; then
    echo "Load average: $(cut -d' ' -f1-3 /proc/loadavg)"
  fi
  # Linux
  if ps -eo pid,user,pcpu,pmem,comm --sort=-pcpu >/dev/null 2>&1; then
    ps -eo pid,user,pcpu,pmem,comm --sort=-pcpu 2>/dev/null | head -6
  # macOS (BSD ps)
  else
    ps -arcwwwxo "pid,user,%cpu,%mem,comm" 2>/dev/null | head -6
  fi
}

puertos_abiertos() {
  if command -v ss >/dev/null 2>&1; then
    ss -tln 2>/dev/null | head -10
  elif command -v netstat >/dev/null 2>&1; then
    netstat -tln 2>/dev/null | head -10
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP -sTCP:LISTEN 2>/dev/null | head -10
  else
    echo "Ninguna herramienta de puertos disponible"
  fi
}

os_info() {
  if [ -f /etc/os-release ]; then
    . /etc/os-release
    echo "SO:     $NAME $VERSION"
  elif command -v sw_vers >/dev/null 2>&1; then
    sw_vers
  else
    uname -a
  fi
  echo "Kernel: $(uname -r)"
  echo "Arch:  $(uname -m)"
}

case "${1:-}" in
  who) quien ;;
  resources) recursos ;;
  top-processes) top_procesos ;;
  listening) puertos_abiertos ;;
  os-info) os_info ;;
  *) uso ;;
esac
