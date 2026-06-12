# Consensus and BFT

> Consensus + BFT: (1) Agreement (all agree+same), (2) Fault tolerance (byzantine+crash), (3) Raft (leader+log), (4) PBFT (practical+3f+1), (5) Quorum (majority+threshold). majority_vote: counts Dict + > n/2. quorum_decide: check size (len<quorum) + majority. PBFT: 3f+1 min nodes + valid = n-f + majority. Raft: leader (leader_id) + all votes [leader] + followers + majority + match leader. Raft vs PBFT: (1) Raft = crash-fault+leader-based+2f+1, (2) PBFT = byzantine+all-to-all+3f+1, (3) Raft = simpler+faster, (4) PBFT = robust+costly. Criterios: Majority = simple+small+no central, Quorum = threshold+subset+configurable, PBFT = byzantine+all-to-all+robust, Raft = leader+crash+log. Decision: simple -> majority, threshold -> quorum, byzantine -> PBFT, leader -> Raft. Frameworks: langchain, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + consensus.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/13
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar majority_vote con threshold > n/2.
- Implementar quorum_decide con check size.
- Implementar pbft_consensus con 3f+1.
- Implementar raft_consensus con leader.
- Diagnosticar Raft vs PBFT.
- Diagnosticar consensus criteria.

## Constrúyelo

```python
def majority_vote(votes):
    if not votes:
        return None, 0
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    sorted_items = sorted(counts.items(), key=lambda kv: -kv[1])
    top_value, top_count = sorted_items[0]
    if top_count > len(votes) / 2:
        return top_value, top_count
    return None, top_count
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: consensus-and-bft
fase: 16
leccion: 14
---

1. majority + quorum.
2. PBFT + Raft.
3. 3f+1.
4. +Production.
```

## Ejercicios

1. **majority**: probar
   threshold.
2. **PBFT**: probar
   byzantine tolerance.
3. **Desafio**: integrar
   con etcd/Consul.

## Lecturas recomendadas

- "Raft Paper" (Ongaro, 2014)
- "PBFT" (Castro, 1999)
- "Distributed Consensus" (Lamport, 1998)

---

> 📚 **Adaptación al español de la lección [Consensus and BFT]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).