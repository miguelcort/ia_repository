# Repo memory and state

> Repo memory + state: (1) Persistent context (disk+DB), (2) Cross-session (multi-run+resume), (3) File-based state (JSON+YAML), (4) Git-tracked (versioned+diff), (5) Structured storage (index+tags). RepoMemory: base_dir (path+mkdir) + index.json (entries+JSON) + _load (read+JSON parse) + _save (write+indent) + add(key, value, tags, ttl) UUID+path+TTL + get(key) entry+TTL check + search(query, tag) filter+match + delete(key) pop+remove file. Ventajas file-based vs in-memory: persistence (disk+survive restart), cross-session (multi-run+resume), git-tracked (versioned+diff+history), audit (who+when), backup (copy+restore). Criterios: Repo = structured+versioned+persistent, In-memory = fast+ephemeral+per-session, Vector DB = semantic+embeddings+similarity. Decision: structured -> repo, fast -> in-memory, semantic -> vector DB, mix -> all. Frameworks: langchain, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + persistence.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/33
**Tiempo estimado:** ~45 minutos

## Objetivos

- Implementar RepoMemory con base_dir + index.json.
- Implementar add con tags + TTL.
- Implementar get con TTL check.
- Implementar search con query + tag.
- Diagnosticar file-based vs in-memory.

## Constrúyelo

```python
class RepoMemory:
    def add(self, key, value, tags=None, ttl=None):
        entry = {
            "key": key,
            "value": value,
            "tags": tags or [],
            "created_at": time.time(),
            "ttl": ttl,
        }
        if ttl is not None:
            entry["expires_at"] = time.time() + ttl
        path = os.path.join(self.base_dir, f"{uuid.uuid4()}.json")
        with open(path, "w") as f:
            json.dump(entry, f, indent=2)
        self.entries[key] = entry
        self._save()
        return entry
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: repo-memory-state
fase: 14
leccion: 34
---

1. RepoMemory + base_dir.
2. add + get + search.
3. TTL + tags.
4. +Production.
```

## Ejercicios

1. **RepoMemory**: probar
   add + get + persistence.
2. **Search**: probar
   query + tag.
3. **Desafio**: integrar
   con vector DB.

## Lecturas recomendadas

- "LangChain: Memory" (LangChain, 2024)
- "File-Based State" (12 Factor App, 2024)
- "Git-Tracked Memory" (Cursor, 2024)

---

> 📚 **Adaptación al español de la lección [Repo Memory and State]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).