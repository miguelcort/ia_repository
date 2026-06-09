# Lección: 07-docker-para-ia
# Fase: 00
# Prerrequisitos: 01-entorno-desarrollo, 06-entornos-python
# Fuentes:
# - Docker docs: https://docs.docker.com/reference/dockerfile/
# - Best practices: https://docs.docker.com/build/building/best-practices/

# Stage 1: builder
FROM python:3.12-slim AS builder

WORKDIR /build

# Copiar requirements primero para cachear
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Stage 2: runtime
FROM python:3.12-slim AS runtime

# Usuario no-root
RUN useradd --create-home --shell /bin/bash app
WORKDIR /app

# Copiar dependencias instaladas desde el builder
COPY --from=builder /install /usr/local

# Copiar codigo
COPY --chown=app:app code/ /app/code/

USER app

# Default: mostrar version
CMD ["python", "code/main.py", "--version"]
