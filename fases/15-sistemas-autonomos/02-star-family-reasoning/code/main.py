"""
Lección: 02-star-family-reasoning
Fase: 15
Star family reasoning: STaR, Quiet-STaR, V-STaR, ReST.
Self-Taught Reasoner. Generate rationales, filter, fine-tune.
+Reasoning +Self-improving.
"""
from __future__ import annotations


def star_generate_rationale(question, base_answer_fn):
    """STaR: generate rationale, then answer."""
    rationale = base_answer_fn(f"Explain step by step: {question}")
    answer = base_answer_fn(f"Given: {rationale}. Answer: {question}")
    return {"rationale": rationale, "answer": answer}


def star_filter_correct(rationales, ground_truth, check_fn):
    """STaR: filter rationales that lead to correct answer."""
    correct = []
    for r in rationales:
        if check_fn(r["answer"], ground_truth):
            correct.append(r)
    return correct


def quiet_star_reasoning(text, base_answer_fn, n_passes=3):
    """Quiet-STaR: parallel reasoning tokens."""
    reasonings = []
    for _ in range(n_passes):
        r = base_answer_fn(f"Think: {text}")
        reasonings.append(r)
    # combine: use longest
    return max(reasonings, key=len)


def v_star_self_consistency(question, base_answer_fn, n_samples=5):
    """V-STaR: verify + self-consistency."""
    samples = [base_answer_fn(question) for _ in range(n_samples)]
    # majority vote
    from collections import Counter
    counts = Counter(samples)
    return counts.most_common(1)[0][0]


def rest_train(steps_data, base_train_fn):
    """ReST: reinforced self-training. Filter correct, augment, train."""
    correct = [d for d in steps_data if d["correct"]]
    # mock training
    return {"trained_on": len(correct), "total": len(steps_data)}


def main() -> int:
    def mock_llm(prompt):
        return f"response to: {prompt[:30]}"
    r = star_generate_rationale("What is 2+2?", mock_llm)
    print(f"STaR: {r}")
    correct = star_filter_correct([r], "4", lambda a, g: g in a)
    print(f"Correct: {len(correct)}")
    quiet = quiet_star_reasoning("What is AI?", mock_llm)
    print(f"Quiet-STaR: {quiet}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())