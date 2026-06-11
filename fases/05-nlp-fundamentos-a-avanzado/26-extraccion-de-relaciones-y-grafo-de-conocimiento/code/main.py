"""
Lección: 26-extraccion-de-relaciones-y-grafo-de-conocimiento
Fase: 05
Prerrequisitos: 25-vinculacion-de-entidades
"""
from __future__ import annotations
import sys
import re
import numpy as np


def entidades_en_texto(texto):
    """Extrae entidades mayusculas (mock de NER)."""
    return re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", texto)


def extraer_relaciones_patron(texto):
    """Extraccion de relaciones por patrones regex.
    Devuelve tuplas (ent1, relacion, ent2)."""
    relaciones = []
    patrones = [
        (r"(\w+) (?:es presidente|fue presidente) de (\w+)", "presidente_de"),
        (r"(\w+) (?:trabaja en|vive en|nacio en) (\w+)", "vive_en"),
        (r"(\w+) (?:y|e) (\w+)", "relacionado_con"),
    ]
    for patron, relacion in patrones:
        for match in re.finditer(patron, texto, re.IGNORECASE):
            ent1, ent2 = match.group(1), match.group(2)
            if ent1[0].isupper() and ent2[0].isupper():
                relaciones.append((ent1, relacion, ent2))
    return relaciones


def clasificar_relacion_mock(ent1, ent2, contexto):
    """Mock: clasifica el tipo de relacion entre dos entidades.
    En produccion: BERT fine-tune en TACRED, FewRel."""
    texto = contexto.lower()
    if any(k in texto for k in ["presidente", "jefe", "director"]):
        return "trabaja_para"
    if any(k in texto for k in ["nacio en", "vive en", "ciudad"]):
        return "ubicado_en"
    if any(k in texto for k in ["fundó", "creó", "esposa de"]):
        return "relacion_personal"
    return "desconocida"


def main() -> int:
    texto = "Maria es presidente de Pedro. Maria vive en Madrid. Pedro trabaja en Google."
    ents = entidades_en_texto(texto)
    print(f"Entidades: {ents}")
    rels = extraer_relaciones_patron(texto)
    print(f"Relaciones (regex): {rels}")
    for e1, e2 in [(r[0], r[2]) for r in rels]:
        rel = clasificar_relacion_mock(e1, e2, texto)
        print(f"  {e1} --{rel}--> {e2}")
    return 0


if __name__ == "__main__":
    sys.exit(main())