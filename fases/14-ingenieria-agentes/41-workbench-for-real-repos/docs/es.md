# Workbench for real repos

> Workbench for real repos: (1) Clone (git+inspect), (2) Structure (tree+dirs), (3) Language stats (ext+count), (4) Find TODOs (TODO+FIXME), (5) Dependency graph (imports+modules). list_files: os.walk (dirpath+dirnames+filenames) + ignore_dirs (.git+node_modules+__pycache__) + max_files cap. language_stats: ext -> lang Dict + count files. find_todos: read line by line (enumerate) + match patterns (TODO/FIXME/XXX/HACK). Tools: directory_structure (max_depth+ignore_dirs+tree), file_size (os.path.getsize), total_size (sum), ignore dirs (Set), max_depth (Limit). Criterios: Workbench = local+fast+simple, GitHub API = remote+metadata+auth, Deep = full AST+slow+rich. Decision: local -> workbench, remote -> GitHub, AST -> deep, mix -> all. Frameworks: git, custom. +Production, +Reliable, +Standard. Hoy: SOTA 2024-25. 2025: +MCP + A2A + native + repos.

**Tipo:** Construir
**Lenguajes:** Python
**Prerrequisitos:** 14/40
**Tiempo estimado:** ~30 minutos

## Objetivos

- Implementar list_files con ignore_dirs + max_files.
- Implementar language_stats con LANG_EXTENSIONS.
- Implementar find_todos con patterns.
- Implementar directory_structure con max_depth.
- Diagnosticar workbench vs GitHub API.

## Constrúyelo

```python
LANG_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    # ... 12+ langs
}
```

## Úsalo

```bash
cd code
python3 main.py
```

## Despliégalo

```markdown
---
name: workbench-real-repos
fase: 14
leccion: 41
---

1. list_files + ignore.
2. language_stats.
3. find_todos.
4. +Production.
```

## Ejercicios

1. **list_files**: probar
   ignore_dirs.
2. **find_todos**: probar
   patterns.
3. **Desafio**: integrar
   con GitHub API.

## Lecturas recomendadas

- "GitHub API" (GitHub, 2024)
- "SLOCCount" (Wheeler, 2024)
- "Tree-sitter: AST" (Tree-sitter, 2024)

---

> 📚 **Adaptación al español de la lección [Workbench for Real Repos]" del currículo [AI Engineering from Scratch](https://github.com/rohitg00/ai-engineering-from-scratch) (Rohit Ghumare, MIT). Implementación y documentación reescritas desde cero. Ver [CREDITS.md](../../../../CREDITS.md).