"""
Lección: 13-preguntas-y-respuestas
Fase: 05
Prerrequisitos: 12-resumen-de-texto
"""
from __future__ import annotations
import sys
import re
import numpy as np


def tokenizar(texto):
    return re.findall(r"\b\w+\b", texto.lower())


class SimpleQASystem:
    """Sistema de QA extractivo simple: encuentra la oracion mas relevante."""
    def __init__(self):
        self.contextos = []

    def fit(self, contextos):
        self.contextos = contextos

    def predecir(self, pregunta, top_k=1):
        """Encuentra la oracion con mas overlap de tokens con la pregunta."""
        q_tokens = set(tokenizar(pregunta))
        # Stopwords mock
        stopwords = {"que", "cual", "quien", "donde", "como", "el", "la", "de"}
        q_tokens -= stopwords
        mejores = []
        for ctx in self.contextos:
            # Split en oraciones
            ors = re.split(r'(?<=[.!?])\s+', ctx)
            for oracion in ors:
                o_tokens = set(tokenizar(oracion)) - stopwords
                if not o_tokens:
                    continue
                overlap = len(q_tokens & o_tokens)
                if q_tokens:
                    score = overlap / len(q_tokens)
                else:
                    score = 0.0
                mejores.append((score, oracion))
        mejores.sort(key=lambda x: -x[0])
        return [ors for score, ors in mejores[:top_k] if score > 0]


def exact_match(pred, gold):
    """EM: 1 si la respuesta predicha coincide exactamente con la gold."""
    pred_norm = pred.strip().lower()
    gold_norm = gold.strip().lower()
    return 1.0 if pred_norm == gold_norm else 0.0


def f1_score(pred, gold):
    """F1 a nivel de tokens entre pred y gold."""
    pred_tokens = tokenizar(pred)
    gold_tokens = tokenizar(gold)
    if not pred_tokens or not gold_tokens:
        return 0.0
    from collections import Counter
    pred_c = Counter(pred_tokens)
    gold_c = Counter(gold_tokens)
    common = pred_c & gold_c
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def main() -> int:
    qa = SimpleQASystem()
    contextos = [
        "Maria vive en Madrid. Pedro trabaja en Barcelona.",
        "El gato come pescado. El perro ladra fuerte.",
    ]
    qa.fit(contextos)
    pregunta = "quien vive en Madrid"
    res = qa.predecir(pregunta, top_k=1)
    print(f"Pregunta: {pregunta}")
    print(f"Respuesta: {res}")
    print(f"EM: {exact_match(res[0] if res else '', 'Maria vive en Madrid')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())