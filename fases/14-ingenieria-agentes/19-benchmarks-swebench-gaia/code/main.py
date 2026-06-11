"""
Lección: 19-benchmarks-swebench-gaia
Fase: 14
Benchmarks: SWE-bench (code), GAIA (general assistant).
+Standard evaluation. Pass@k, success rate, task completion.
"""
from __future__ import annotations


def swebench_score(predictions, ground_truth, k_values=(1, 5, 10)):
    """SWE-bench style scoring: pass@k con k values."""
    scores = {}
    n = len(ground_truth)
    if n == 0:
        return scores
    correct = [i for i, (p, g) in enumerate(zip(predictions, ground_truth)) if p == g]
    for k in k_values:
        # pass@k
        if k > n:
            scores[f"pass@{k}"] = 0.0
            continue
        scores[f"pass@{k}"] = len(correct) / n
    return scores


def gaia_score(predictions, ground_truth, levels=("level1", "level2", "level3")):
    """GAIA scoring: success per level."""
    scores = {}
    for level in levels:
        idx = [i for i, g in enumerate(ground_truth) if g.get("level") == level]
        if not idx:
            continue
        correct = sum(1 for i in idx if predictions[i] == ground_truth[i]["answer"])
        scores[level] = correct / len(idx)
    return scores


def success_rate(predictions, ground_truth):
    """Overall success rate."""
    if not ground_truth:
        return 0.0
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    return correct / len(ground_truth)


def main() -> int:
    preds = ["a", "b", "a", "c"]
    truth = ["a", "b", "c", "c"]
    sw = swebench_score(preds, truth)
    print(f"SWE-bench: {sw}")
    truth_gaia = [
        {"level": "level1", "answer": "a"},
        {"level": "level1", "answer": "b"},
        {"level": "level2", "answer": "c"},
        {"level": "level3", "answer": "a"},
    ]
    preds_gaia = ["a", "b", "c", "x"]
    ga = gaia_score(preds_gaia, truth_gaia)
    print(f"GAIA: {ga}")
    print(f"Success: {success_rate(preds, truth):.2f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())