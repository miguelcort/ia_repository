"""
Lección: 05-eagle3-speculative-decoding
Fase: 17
EAGLE3 speculative decoding: draft
model predicts multiple tokens, target
verifies in parallel. Speedup without
quality loss.
"""
from __future__ import annotations
import random


class SpeculativeDecoder:
    def __init__(self, draft_model_size=1, target_model_size=70,
                 num_speculative=5, accept_rate=0.7, seed=None):
        self.draft_size = draft_model_size
        self.target_size = target_model_size
        self.num_speculative = num_speculative
        self.accept_rate = accept_rate
        if seed is not None:
            random.seed(seed)

    def draft_tokens(self, num=None):
        n = num or self.num_speculative
        return [f"draft_{i}" for i in range(n)]

    def verify(self, drafts):
        """Return accepted count based on accept_rate."""
        accepted = 0
        for d in drafts:
            if random.random() < self.accept_rate:
                accepted += 1
        return accepted

    def step(self):
        drafts = self.draft_tokens()
        accepted = self.verify(drafts)
        return drafts, accepted

    def speedup_factor(self):
        """Estimate: 1 draft + 1 target verify (parallel) vs sequential target."""
        sequential_tokens = self.num_speculative
        parallel_tokens = 1 + self.accept_rate * self.num_speculative
        if parallel_tokens == 0:
            return 1.0
        return sequential_tokens / parallel_tokens


def main() -> int:
    dec = SpeculativeDecoder(accept_rate=0.7, seed=42)
    drafts, accepted = dec.step()
    print(f"Accepted: {accepted}/{len(drafts)}, speedup: {dec.speedup_factor():.2f}x")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())