"""
Lección: 10-evaluacion
Fase: 11
Evaluation: A/B tests, human eval, LM-as-judge, RAGAS, custom.
"""
from __future__ import annotations
import sys
import numpy as np


def lm_judge_score(prompt, response, reference):
    """Mock LM-as-judge: 0-1 score basado en overlap con reference."""
    if not response or not reference:
        return 0.0
    p_tokens = set(prompt.split())
    r_tokens = set(response.split())
    ref_tokens = set(reference.split())
    if not ref_tokens:
        return 0.0
    overlap = len(r_tokens & ref_tokens)
    return min(1.0, overlap / len(ref_tokens))


def ab_test_se(n_a, n_b, alpha=0.05):
    """A/B test significance. Returns p-value (mock)."""
    # Mock: returns random p-value
    from math import sqrt
    if n_a + n_b == 0:
        return 1.0
    # Simple z-test mock
    p_pool = 0.5
    diff = abs(n_a - n_b) / max(n_a + n_b, 1)
    return max(0.0, 1.0 - diff * 10)


def precision_at_k(relevant_docs, retrieved_docs, k=10):
    """Precision@k: relevant in top-k / k."""
    top_k = retrieved_docs[:k]
    relevant_in_top = len(set(top_k) & set(relevant_docs))
    return relevant_in_top / k


def recall_at_k(relevant_docs, retrieved_docs, k=10):
    """Recall@k: relevant in top-k / total relevant."""
    if not relevant_docs:
        return 0.0
    top_k = retrieved_docs[:k]
    relevant_in_top = len(set(top_k) & set(relevant_docs))
    return relevant_in_top / len(relevant_docs)


def bleu_score(reference, candidate, n=2):
    """Mock BLEU score (n-gram precision)."""
    if not reference or not candidate:
        return 0.0
    ref_words = reference.split()
    cand_words = candidate.split()
    if len(cand_words) < n:
        return 0.0
    # Simple n-gram precision
    matches = 0
    for i in range(len(cand_words) - n + 1):
        ngram = " ".join(cand_words[i:i + n])
        if ngram in " ".join(ref_words):
            matches += 1
    return matches / max(len(cand_words) - n + 1, 1)


def main() -> int:
    score = lm_judge_score("What is AI?", "AI is artificial intelligence.", "AI is artificial intelligence")
    print(f"LM judge score: {score:.2f}")
    p = ab_test_se(60, 50)
    print(f"A/B p-value: {p:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())