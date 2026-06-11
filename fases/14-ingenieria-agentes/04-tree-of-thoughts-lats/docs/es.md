# Tree of thoughts LATS

> ToT (Yao 2023, Tree of Thoughts) y LATS (Zhou 2023, Language Agent Tree Search): tree search para reasoning + planning. ToT: BFS/DFS, expand + evaluate, +simple, +exploration. LATS: MCTS-based con UCB1, +planning, +search, +MCTS. TreeNode: state + parent + children + value + visits. bfs_expand (max_depth, expand_fn, evaluate_fn) y mcts (selection UCB1 + expansion + simulation + backprop). UCB1: value/visits + c * sqrt(log(visits_parent)/visits). +Reasoning, +Exploration, +Planning, +Production, +Search, +Exploitation. Variants: ToT, LATS, A*, BFS, DFS, MCTS. Frameworks: langchain, openai, anthropic. +Production: standard 2024-25. +Use cases: agent, planning, problem solving, games. Decision: simple -> BFS/ToT, planning -> LATS, optimal -> A*, production -> LATS. Trade-offs: cada uno + specialty, ToT + simple, LATS + planning. 2025: +MCP + A2A + native + variants.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/01
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar TreeNode class.
- Implementar bfs_expand.
- Implementar ucb_select y mcts.
- Diagnosticar ToT vs LATS vs A*.
- Diagnosticar MCTS variants.

## Constrúyelo

```python
def ucb_select(node, c=1.414):
    if node.visits == 0:
        return float("inf")
    return node.value / node.visits + c * math.sqrt(math.log(max(1, node.parent.visits)) / node.visits)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: tot-lats
fase: 14
leccion: 04
---

1. ToT BFS/DFS.
2. LATS MCTS.
3. UCB1.
4. +Reasoning.
5. +Planning.
```

## Ejercicios

1. **ToT**: implementar
   ToT custom con BFS.
2. **LATS**: probar
   LATS con MCTS.
3. **Desafio**: full
   reasoning agent.

## Lecturas recomendadas

- "Tree of Thoughts: Deliberate Problem Solving with Large Language Models" (Yao et al., 2023)
- "Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models" (Zhou et al., 2023)
- "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (Silver et al., 2017)
- "A Survey of Monte Carlo Tree Search Methods" (Browne et al., 2012)

---

> 📚 **Adaptación al español de la lección [Tree of Thoughts LATS]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).