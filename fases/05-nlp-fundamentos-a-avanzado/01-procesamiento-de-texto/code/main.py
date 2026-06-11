"""
Lección: 01-procesamiento-de-texto
Fase: 05
Prerrequisitos: 02-fundamentos-ml/02-modelos-lineales-y-regresion-logistica
"""
from __future__ import annotations
import sys
import re
import numpy as np


def normalizar(texto, lowercase=True, quitar_acentos=False):
    """Normalizacion basica: lowercase, opcionalmente quitar acentos."""
    if lowercase:
        texto = texto.lower()
    if quitar_acentos:
        # Mapeo manual de acentos comunes
        reemplazos = {
            "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
            "à": "a", "è": "e", "ì": "i", "ò": "o", "ù": "u",
            "ñ": "n", "ü": "u",
        }
        for orig, repl in reemplazos.items():
            texto = texto.replace(orig, repl)
    return texto


def tokenizar(texto, patron=r"\b\w+\b"):
    """Tokeniza usando regex. Devuelve lista de tokens."""
    return re.findall(patron, texto.lower())


def quitar_stopwords(tokens, stopwords=None):
    """Elimina stopwords (palabras vacias)."""
    if stopwords is None:
        stopwords = {"el", "la", "los", "las", "un", "una", "de", "del",
                     "a", "y", "o", "en", "por", "para", "con", "sin",
                     "the", "a", "an", "of", "to", "and", "or", "in"}
    return [t for t in tokens if t not in stopwords]


def stemming(tokens):
    """Stemming super-simplificado: corta sufijos comunes.
    En la practica, usar Porter o Snowball."""
    sufijos = ["ando", "iendo", "ar", "er", "ir", "ado", "ido", "mente", "s"]
    stems = []
    for t in tokens:
        for s in sufijos:
            if t.endswith(s) and len(t) - len(s) >= 3:
                t = t[: -len(s)]
                break
        stems.append(t)
    return stems


def n_gramas(tokens, n=2):
    """Genera n-gramas de una secuencia."""
    if n < 1 or len(tokens) < n:
        return []
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def main() -> int:
    texto = "El gato negro salta sobre el perro. Los perros ladran."
    print(f"Original: {texto}")
    norm = normalizar(texto, lowercase=True, quitar_acentos=True)
    print(f"Normalizado: {norm}")
    tokens = tokenizar(norm)
    print(f"Tokens: {tokens}")
    tokens_limpios = quitar_stopwords(tokens)
    print(f"Sin stopwords: {tokens_limpios}")
    stems = stemming(tokens_limpios)
    print(f"Stems: {stems}")
    bigramas = n_gramas(tokens, n=2)
    print(f"Bigramas: {bigramas}")
    return 0


if __name__ == "__main__":
    sys.exit(main())