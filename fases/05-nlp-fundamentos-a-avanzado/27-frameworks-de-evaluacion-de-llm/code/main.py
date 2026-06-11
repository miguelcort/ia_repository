"""
Lección: 27-frameworks-de-evaluacion-de-llm
Fase: 05
Prerrequisitos: 26-extraccion-de-relaciones-y-grafo-de-conocimiento
"""
from __future__ import annotations
import sys
import numpy as np


def exact_match(pred, gold):
    return 1.0 if pred.strip().lower() == gold.strip().lower() else 0.0


def f1_token(pred, gold):
    """F1 a nivel de tokens."""
    p_tokens = set(pred.lower().split())
    g_tokens = set(gold.lower().split())
    if not p_tokens or not g_tokens:
        return 0.0
    common = p_tokens & g_tokens
    if not common:
        return 0.0
    p = len(common) / len(p_tokens)
    r = len(common) / len(g_tokens)
    return 2 * p * r / (p + r)


def rouge_l(pred, gold):
    """ROUGE-L simplificado."""
    p_tokens = pred.lower().split()
    g_tokens = gold.lower().split()
    if not p_tokens or not g_tokens:
        return 0.0
    # LCS
    m, n = len(p_tokens), len(g_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p_tokens[i - 1] == g_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    p = lcs / m
    r = lcs / n
    return 2 * p * r / (p + r)


def llm_as_judge_mock(prompt, response, reference):
    """Mock: LLM-as-judge con scoring 1-5. En produccion, GPT-4 o Claude."""
    rng = np.random.default_rng(hash(response + reference) % 2**32)
    # Heuristica mock: si hay palabras en comun -> score alto
    p_words = set(prompt.lower().split())
    r_words = set(reference.lower().split())
    overlap = len(p_words & r_words) / max(1, len(r_words))
    return int(1 + overlap * 4)


def faithfulness_score(respuesta, contexto):
    """Medicion simple de faithfulness: % de tokens de la respuesta en el contexto."""
    r = set(respuesta.lower().split())
    c = set(contexto.lower().split())
    if not r:
        return 0.0
    return len(r & c) / len(r)


def main() -> int:
    pred = "Maria vive en Madrid"
    gold = "Maria vive en Madrid capital"
    print(f"EM: {exact_match(pred, gold)}")
    print(f"F1 token: {f1_token(pred, gold):.3f}")
    print(f"ROUGE-L: {rouge_l(pred, gold):.3f}")
    score = llm_as_judge_mock("donde vive Maria?", pred, gold)
    print(f"LLM-as-judge: {score}/5")
    print(f"Faithfulness: {faithfulness_score(pred, 'Maria vive en Madrid capital de Espana'):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())