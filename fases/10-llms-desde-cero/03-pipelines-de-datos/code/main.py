"""
Lección: 03-pipelines-de-datos
Fase: 10
Data pipelines para pre-training: download, filter, dedup, tokenize, pack.
Datasets: Common Crawl, RefinedWeb, FineWeb, RedPajama, The Pile.
"""
from __future__ import annotations
import sys
import re
import hashlib
from collections import Counter


def normalize_text(text):
    """Normalizacion basica: lowercase, strip whitespace, NFC unicode."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def quality_filter(text, min_length=100, max_line_breaks=10):
    """Filtros de calidad: longitud, line breaks, ratios."""
    if len(text) < min_length:
        return False
    if text.count("\n") > max_line_breaks:
        return False
    # Ratio de alphanumeric
    alnum = sum(1 for c in text if c.isalnum())
    if alnum / len(text) < 0.5:
        return False
    return True


def language_id_mock(text, target_lang="en"):
    """Mock language ID: en textos cortos, usar heuristica.
    Real: fastText, langdetect, CLD3.
    """
    # Mock: solo aceptar si tiene suficiente latin
    if target_lang == "en":
        common_words = {"the", "and", "or", "is", "a", "an", "to", "in", "of", "for"}
        words = set(text.lower().split())
        return len(words & common_words) > 0
    return True


def dedup_hashing(text, prefix_len=64):
    """Hash-based deduplication: compute hash de primeros N chars."""
    prefix = text[:prefix_len]
    return hashlib.md5(prefix.encode()).hexdigest()


def dedup_hashes(docs, prefix_len=64):
    """Compute hashes y devuelve unique docs (primeras ocurrencias)."""
    seen = set()
    unique = []
    for doc in docs:
        h = dedup_hashing(doc, prefix_len)
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique


def pii_filter(text):
    """Mock PII filter: detecta y reemplaza emails y telefonos.
    """
    text = re.sub(r"\b[\w.-]+@[\w.-]+\.\w+\b", "[EMAIL]", text)
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "[PHONE]", text)
    return text


def pack_sequences(token_ids, max_len, eos_token):
    """Pack tokens en secuencias de max_len con EOS al final."""
    sequences = []
    current = []
    for tid in token_ids:
        current.append(tid)
        if len(current) >= max_len:
            current.append(eos_token)
            sequences.append(current)
            current = []
    if current:
        # Pad final sequence
        while len(current) < max_len:
            current.append(0)
        sequences.append(current)
    return sequences


def main() -> int:
    docs = [
        "The quick brown fox jumps over the lazy dog. " * 5,
        "Hola mundo, esto es un test.",  # Muy corto
        "The quick brown fox jumps over the lazy dog. " * 5,  # dup
        "Email me at test@example.com or call 555-123-4567.",
    ]
    # Filter
    filtered = [d for d in docs if quality_filter(d)]
    print(f"Filtered: {len(filtered)}/{len(docs)}")
    # Dedup
    unique = dedup_hashes(filtered, prefix_len=20)
    print(f"Unique: {len(unique)}")
    # PII
    pii_cleaned = [pii_filter(d) for d in unique]
    print(f"PII cleaned: {pii_cleaned[-1][:50]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())