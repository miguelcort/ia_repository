# SGLang RadixAttention

> RadixAttention: (1) Tree (prefix cache+radix), (2) Fast (lookup+O(k)), (3) Structured (JSON+grammar), (4) Share (across requests+reuse), (5) LRU (eviction+memory). RadixNode: key+children+last_access+value+is_leaf+touch(). RadixTree: root+insert(tokens, value)+lookup(tokens)+evict_lru(current_time)+size+_walk_leaves. Ventajas RadixAttention vs linear: O(k) fast+tree, share across+reuse, LRU evict+memory, structured JSON+grammar, multi-tenant isolated+shared. Criterios: Radix = shared+multi-user+structured, Simple = per-req+single user+custom, None = cold+cheap+simple. Decision: shared -> radix, per-req -> simple, cold -> none, mix -> radix+simple. Frameworks: sglang, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + radix.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 17/05
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar RadixNode con key + children + last_access + value.
- Implementar RadixTree con root + insert + lookup.
- Implementar evict_lru con time threshold.
- Diagnosticar advantages.
- Diagnosticar Radix vs simple vs no cache.

## Constrúyelo

```python
class RadixTree:
    def lookup(self, tokens):
        node = self.root
        matched = []
        for token in tokens:
            if token in node.children:
                node = node.children[token]
                node.touch()
                matched.append(token)
            else:
                break
        return matched
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: sglang-radixattention
fase: 17
leccion: 06
---

1. RadixNode + RadixTree.
2. insert + lookup.
3. evict_lru.
4. +Production.
```

## Ejercicios

1. **RadixTree**: probar
   insert + lookup.
2. **Evict**: probar
   LRU.
3. **Desafio**: integrar
   con SGLang.

## Lecturas recomendadas

- "SGLang" (Zheng, 2024)
- "RadixAttention" (Zheng, 2024)
- "Prefix Cache" (Kwon, 2023)

---

> 📚 **Adaptación al español de la lección [SGLang RadixAttention]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).