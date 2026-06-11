# RL para juegos

> Hitos: AlphaGo (2016, MCTS+CNN+RL, vence Lee Sedol), AlphaZero (2017, self-play sin data humana, domina Go/chess/shogi), AlphaStar (2019, StarCraft 2 Grandmaster), OpenAI Five (2018, Dota 2 5v5), MuZero (2019, MCTS en latent model), Pluribus (2019, poker). MCTS+NN, self-play, league training, EfficientZero. Frontier: foundation models + games, agentic AI, world models (GameNGen, Genie 2, Cosmos).

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 09/08-ppo
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar MCTS PUCT select.
- Implementar MCTS backprop.
- Implementar self-play iteration.
- Comparar AlphaGo, AlphaZero, MuZero, AlphaStar.

## Constrúyelo

```python
def mcts_select(node, c_puct=1.0):
    best_score = -np.inf
    for action, child in node.children.items():
        u = child.q_value + c_puct * child.prior * np.sqrt(node.n_visits) / (1 + child.n_visits)
        if u > best_score:
            best_score = u
            best_child = child
    return best_child
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: prompt-rl-games
fase: 09
leccion: 12
---

1. AlphaZero: MCTS+NN+self-play.
2. MuZero: MCTS en latent.
3. AlphaStar: league training.
4. PUCT selection.
5. Frontier: world models.
```

## Ejercicios

1. **Tic-tac-toe**: implementar
   self-play AlphaZero-style.
2. **Connect Four**: resolver con
   MCTS + NN.
3. **Desafio**: MuZero en Atari.

## Lecturas recomendadas

- "Mastering the Game of Go with Deep Neural Networks and Tree Search" (Silver et al., 2016)
- "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (Silver et al., 2017)
- "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (Schrittwieser et al., 2020)
- "Grandmaster Level in StarCraft II Using Multi-Agent Reinforcement Learning" (Vinyals et al., 2019)

---

> 📚 **Adaptación al español** de la lección "[RL for Games]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).