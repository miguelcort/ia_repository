"""
Lección: 14-speculative-decoding-server
Fase: 19
Capstone de ingeniería AI: 14 Speculative Decoding Server.
"""
from __future__ import annotations
import sys

def speculative_decode(draft_model, target_model, prompt, k=5):
    """Speculative decoding: draft K, target verify."""
    tokens = tokenize(prompt)
    while not_finished(tokens):
        draft = draft_model.generate(tokens, k=k)
        target_logits = target_model(draft)
        accepted = match_prefix(draft, target_logits)
        tokens.extend(draft[:accepted])
    return tokens



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
