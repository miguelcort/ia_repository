# Hierarchical architecture

> Hierarchical: (1) Tree (parent+children+levels), (2) Delegation (up+down+pass), (3) Levels (manager+lead+worker+command), (4) Decomposition (top-down+sub-tasks), (5) Synthesis (aggregate+bottom-up). HierarchicalNode: parent (Ref)+children (List)+add_child(child) set parent+set level+is_leaf() no children+delegate(task) leaf->_execute, internal->children+_aggregate, _execute (leaf:role:action), _aggregate (agg:role:results), tree(depth) prefix+recursive. Roles jerarquicos: Manager (root+level 0+decide), Tech lead (level 1+technical), Product lead (level 1+product), Developer (level 2+code), Tester (level 2+test), Reviewer (level 2+review). Criterios: Hierarchical = orgs+levels+tree, Supervisor = central+simple+single, Flat = peer-to-peer+no central+simple, A2A = modern+standard+Google. Decision: orgs -> hierarchical, simple -> supervisor, peer -> flat, modern -> A2A. Frameworks: langgraph, anthropic, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + hierarchical.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 16/05
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar HierarchicalNode con parent + children + level.
- Implementar add_child + is_leaf.
- Implementar delegate recursivo.
- Implementar _execute + _aggregate.
- Implementar tree() recursivo.
- Diagnosticar roles.

## Constrúyelo

```python
class HierarchicalNode:
    def delegate(self, task):
        if self.is_leaf():
            return self._execute(task)
        sub_results = [child.delegate(task) for child in self.children]
        return self._aggregate(sub_results)
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: hierarchical-architecture
fase: 16
leccion: 06
---

1. HierarchicalNode + tree.
2. delegate + aggregate.
3. Roles: manager, leads,
   developers.
4. +Production.
```

## Ejercicios

1. **HierarchicalNode**: probar
   add_child + tree.
2. **delegate**: probar
   leaf + aggregate.
3. **Desafio**: integrar
   con LangGraph.

## Lecturas recomendadas

- "Multi-Agent Hierarchies" (LangChain, 2024)
- "Society of Mind" (Minsky, 1986)
- "Organizational Design" (Galbraith, 2014)

---

> 📚 **Adaptación al español de la lección [Hierarchical Architecture]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).