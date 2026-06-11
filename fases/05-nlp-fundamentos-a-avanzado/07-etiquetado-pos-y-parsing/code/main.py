"""
Lección: 07-etiquetado-pos-y-parsing
Fase: 05
Prerrequisitos: 06-reconocimiento-de-entidades
"""
from __future__ import annotations
import sys
import re


def tokenizar(texto):
    return re.findall(r"\b\w+\b", texto.lower())


def pos_tag_mock(tokens):
    """Mock: asigna POS tag a cada token.
    Reglas simples: 'el/la' -> DET, 'gato/perro' -> NOUN, 'come/salta' -> VERB, etc."""
    det = {"el", "la", "los", "las", "un", "una", "unos", "unas"}
    verbos = {"come", "salta", "ladra", "canta", "corre", "vuela", "es"}
    nombres = {"gato", "perro", "pajaro", "nino", "casa", "libro"}
    adj = {"negro", "blanco", "rojo", "grande", "pequeno", "bonito"}
    adv = {"muy", "poco", "bien", "mal", "rapido", "lento"}
    tags = []
    for t in tokens:
        if t in det:
            tags.append("DET")
        elif t in verbos:
            tags.append("VERB")
        elif t in nombres:
            tags.append("NOUN")
        elif t in adj:
            tags.append("ADJ")
        elif t in adv:
            tags.append("ADV")
        else:
            tags.append("NOUN")  # default
    return tags


def dep_parse_mock(tokens, tags):
    """Mock: parsing de dependencias basico. root = primer verbo, demas nouns son sus hijos."""
    # Encuentra el root (primer VERB o primer token)
    root = -1
    for i, tag in enumerate(tags):
        if tag == "VERB":
            root = i
            break
    if root == -1:
        root = 0
    # Asigna relaciones simples
    edges = [(root, "root")]
    for i, (t, tag) in enumerate(zip(tokens, tags)):
        if i == root:
            continue
        if tag == "DET":
            # Determinante va con el siguiente noun
            for j in range(i + 1, len(tokens)):
                if tags[j] == "NOUN":
                    edges.append((j, i, "det"))
                    break
        elif tag in ("ADJ", "NOUN"):
            # Adj o noun no-root van al root o al noun anterior
            for j in range(i - 1, -1, -1):
                if tags[j] in ("NOUN", "VERB"):
                    edges.append((j, i, "mod" if tag == "ADJ" else "conj" if j == root else "nmod"))
                    break
    return edges


def main() -> int:
    tokens = tokenizar("el gato negro come pescado")
    tags = pos_tag_mock(tokens)
    print(f"Tokens: {tokens}")
    print(f"POS tags: {tags}")
    edges = dep_parse_mock(tokens, tags)
    print(f"Deps: {edges}")
    return 0


if __name__ == "__main__":
    sys.exit(main())