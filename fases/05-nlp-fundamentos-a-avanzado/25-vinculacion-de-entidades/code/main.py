"""
Lección: 25-vinculacion-de-entidades
Fase: 05
Prerrequisitos: 24-resolucion-de-coreferencias
"""
from __future__ import annotations
import sys
import numpy as np


class KnowledgeBase:
    """Mock: base de conocimiento de entidades (Wikidata-like)."""
    def __init__(self):
        self.entidades = {
            "Q76": {"name": "Barack Obama", "description": "44th President of the US", "aliases": ["Obama", "President Obama"]},
            "Q5": {"name": "United States", "description": "Country in North America", "aliases": ["USA", "US", "America"]},
            "Q64": {"name": "Berlin", "description": "Capital of Germany", "aliases": ["Berlin city"]},
        }

    def get(self, qid):
        return self.entidades.get(qid)

    def search(self, texto, top_k=3):
        """Mock: matching por substring."""
        texto_lower = texto.lower()
        scores = []
        for qid, ent in self.entidades.items():
            score = 0
            for alias in [ent["name"]] + ent.get("aliases", []):
                if alias.lower() in texto_lower:
                    score += 1
            scores.append((score, qid))
        scores.sort(reverse=True)
        return [qid for s, qid in scores[:top_k] if s > 0]


def entity_linking(mention, kb, embeddings=None):
    """Vincula una mencion a la entidad correcta en la KB.
    Estrategia: exact match primero, luego fuzzy con embeddings.
    """
    # Exact match
    for qid, ent in kb.entidades.items():
        if mention.lower() == ent["name"].lower():
            return qid
        if mention.lower() in [a.lower() for a in ent.get("aliases", [])]:
            return qid
    # Partial match
    for qid, ent in kb.entidades.items():
        if mention.lower() in ent["name"].lower():
            return qid
    return None


def linkear_texto(texto, kb):
    """Encuentra entidades en el texto y las vincula a la KB.
    Mock: solo mayusculas con multiple palabras."""
    import re
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", texto)
    resultados = []
    for c in candidates:
        qid = entity_linking(c, kb)
        if qid:
            resultados.append((c, qid, kb.get(qid)["name"]))
    return resultados


def main() -> int:
    kb = KnowledgeBase()
    texto = "Barack Obama fue presidente de United States. Vive en Washington DC."
    resultados = linkear_texto(texto, kb)
    print(f"Entidades vinculadas: {resultados}")
    return 0


if __name__ == "__main__":
    sys.exit(main())