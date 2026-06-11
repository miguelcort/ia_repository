"""
Lección: 06-reconocimiento-de-entidades
Fase: 05
Prerrequisitos: 05-analisis-de-sentimiento
"""
from __future__ import annotations
import sys
import re
import numpy as np


def tokenizar(texto):
    return re.findall(r"\b\w+\b", texto)


def bio_tags(tokens):
    """Convierte tokens a tags BIO mock. Asume 'Maria' es persona, 'Madrid' lugar, etc."""
    # Lexicon mock
    lexicon = {
        "maria": "B-PER", "juan": "B-PER", "pedro": "B-PER",
        "madrid": "B-LOC", "barcelona": "B-LOC", "lima": "B-LOC",
        "google": "B-ORG", "microsoft": "B-ORG", "openai": "B-ORG",
    }
    tags = []
    for t in tokens:
        if t.lower() in lexicon:
            tags.append(lexicon[t.lower()])
        else:
            tags.append("O")
    return tags


def extraer_entidades(tokens, tags):
    """Extrae entidades: lista de (tipo, span, texto) desde BIO tags."""
    entidades = []
    actual_tipo = None
    inicio = None
    for i, (t, tag) in enumerate(zip(tokens, tags)):
        if tag.startswith("B-"):
            if actual_tipo is not None:
                entidades.append((actual_tipo, (inicio, i - 1), " ".join(tokens[inicio:i])))
            actual_tipo = tag[2:]
            inicio = i
        elif tag.startswith("I-"):
            if actual_tipo != tag[2:]:
                # I- sin B-: tratar como B-
                if actual_tipo is not None:
                    entidades.append((actual_tipo, (inicio, i - 1), " ".join(tokens[inicio:i])))
                actual_tipo = tag[2:]
                inicio = i
        else:  # O
            if actual_tipo is not None:
                entidades.append((actual_tipo, (inicio, i - 1), " ".join(tokens[inicio:i])))
                actual_tipo = None
                inicio = None
    if actual_tipo is not None:
        entidades.append((actual_tipo, (inicio, len(tokens) - 1), " ".join(tokens[inicio:])))
    return entidades


def span_f1(pred_entities, gold_entities):
    """F1 de span-based NER. Match exacto por (tipo, span)."""
    pred_set = set((t, s) for t, s, _ in pred_entities)
    gold_set = set((t, s) for t, s, _ in gold_entities)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main() -> int:
    texto = "Maria trabaja en Google en Madrid"
    tokens = tokenizar(texto)
    tags = bio_tags(tokens)
    print(f"Tokens: {tokens}")
    print(f"Tags: {tags}")
    entidades = extraer_entidades(tokens, tags)
    print(f"Entidades: {entidades}")
    return 0


if __name__ == "__main__":
    sys.exit(main())