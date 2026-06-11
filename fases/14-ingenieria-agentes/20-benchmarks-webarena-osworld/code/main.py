"""
Lección: 20-benchmarks-webarena-osworld
Fase: 14
WebArena (CMU 2023) y OSWorld (2024): realistic web + OS
benchmarks. Computer use + navigation + realistic tasks.
+Standard evaluation. Browser + OS environments.
"""
from __future__ import annotations


def webarena_score(predictions, ground_truth, sites=("shopping", "reddit", "gitlab", "maps")):
    """WebArena scoring: success per site."""
    scores = {}
    for site in sites:
        idx = [i for i, g in enumerate(ground_truth) if g.get("site") == site]
        if not idx:
            continue
        correct = sum(1 for i in idx if predictions[i] == ground_truth[i]["answer"])
        scores[site] = correct / len(idx)
    return scores


def osworld_score(predictions, ground_truth, apps=("chrome", "vscode", "gimp", "terminal")):
    """OSWorld scoring: success per app."""
    scores = {}
    for app in apps:
        idx = [i for i, g in enumerate(ground_truth) if g.get("app") == app]
        if not idx:
            continue
        correct = sum(1 for i in idx if predictions[i] == ground_truth[i]["answer"])
        scores[app] = correct / len(idx)
    return scores


def task_completion_rate(predictions, ground_truth, partial_credit=True):
    """Task completion rate. partial_credit: 0.5 for partial."""
    if not ground_truth:
        return 0.0
    total = 0
    for p, g in zip(predictions, ground_truth):
        if p == g.get("answer"):
            total += 1.0
        elif partial_credit and g.get("partial") and p:
            # mock: any non-empty partial
            total += 0.5
    return total / len(ground_truth)


def main() -> int:
    preds = ["a", "b", "c", "a"]
    truth_wa = [
        {"site": "shopping", "answer": "a"},
        {"site": "reddit", "answer": "b"},
        {"site": "shopping", "answer": "x"},
        {"site": "maps", "answer": "a"},
    ]
    wa = webarena_score(preds, truth_wa)
    print(f"WebArena: {wa}")
    truth_os = [
        {"app": "chrome", "answer": "a"},
        {"app": "vscode", "answer": "b"},
    ]
    osw = osworld_score(["a", "x"], truth_os)
    print(f"OSWorld: {osw}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())