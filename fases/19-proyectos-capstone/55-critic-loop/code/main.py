"""
Lección: 55-critic-loop
Fase: 19
Capstone de ingeniería AI: 55 Critic Loop.
"""
from __future__ import annotations
import sys

def critic_loop(generator, critic, prompt, max_iters=3,
              threshold=0.8):
    output = generator(prompt)
    for i in range(max_iters):
        score, critique = critic(output)
        if score >= threshold:
            return output
        output = generator(f"Refine: {prompt}\nCritique: {critique}")
    return output



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
