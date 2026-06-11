"""
Lección: 23-estrategias-de-chunking-y-rag
Fase: 05
Prerrequisitos: 22-modelos-de-embedding-a-profundidad
"""
from __future__ import annotations
import sys
import re


def chunking_fijo(texto, chunk_size=200, overlap=50):
    """Divide el texto en chunks de tamano fijo con overlap."""
    palabras = texto.split()
    chunks = []
    i = 0
    while i < len(palabras):
        chunk = palabras[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        if i + chunk_size >= len(palabras):
            break
        i += chunk_size - overlap
    return chunks


def chunking_oraciones(texto, max_chars=500):
    """Divide respetando fronteras de oraciones."""
    oraciones = re.split(r'(?<=[.!?])\s+', texto)
    chunks = []
    actual = ""
    for oracion in oraciones:
        if len(actual) + len(oracion) > max_chars and actual:
            chunks.append(actual.strip())
            actual = oracion
        else:
            actual += " " + oracion
    if actual:
        chunks.append(actual.strip())
    return chunks


def chunking_semantico_mock(texto, n_chunks=3):
    """Mock: divide en n_chunks aproximadamente iguales."""
    palabras = texto.split()
    tam = max(1, len(palabras) // n_chunks)
    chunks = []
    for i in range(n_chunks):
        chunk = palabras[i * tam:(i + 1) * tam]
        if chunk:
            chunks.append(" ".join(chunk))
    return chunks


def overlap_palabras(chunk_a, chunk_b):
    """% de palabras que aparecen en ambos chunks."""
    pa = set(chunk_a.lower().split())
    pb = set(chunk_b.lower().split())
    if not pa or not pb:
        return 0.0
    return len(pa & pb) / len(pa | pb)


def main() -> int:
    texto = (
        "Maria vive en Madrid. Pedro trabaja en Barcelona. "
        "El gato come pescado. El perro ladra fuerte. "
        "Los pajaros vuelan alto en el cielo. El sol brilla en verano. "
        "La luna sale de noche. Las estrellas iluminan el firmamento."
    )
    chunks_fijos = chunking_fijo(texto, chunk_size=10, overlap=3)
    print(f"Fijos ({len(chunks_fijos)}):")
    for c in chunks_fijos:
        print(f"  - {c[:50]}")
    chunks_or = chunking_oraciones(texto, max_chars=50)
    print(f"\nOraciones ({len(chunks_or)}):")
    for c in chunks_or:
        print(f"  - {c[:50]}")
    if len(chunks_fijos) >= 2:
        ov = overlap_palabras(chunks_fijos[0], chunks_fijos[1])
        print(f"\nOverlap: {ov:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())