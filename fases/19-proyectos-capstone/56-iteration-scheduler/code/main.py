"""
Lección: 56-iteration-scheduler
Fase: 19
Capstone de ingeniería AI: 56 Iteration Scheduler.
"""
from __future__ import annotations
import sys

import optuna


def objective(trial):
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    bs = trial.suggest_categorical("bs", [16, 32, 64])
    return train_and_eval(lr, bs)


def run_search(n_trials=50):
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params



def main() -> int:
    """Demo auto-terminal. Imprime resumen."""
    print(f"=== {slug} ===")
    print(f"Python {sys.version.split()[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
