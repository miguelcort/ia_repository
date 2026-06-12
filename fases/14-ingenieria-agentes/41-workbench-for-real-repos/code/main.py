"""
Lección: 41-workbench-for-real-repos
Fase: 14
Workbench for real repos: clone,
inspect structure, language stats,
find TODOs, dependency graph.
"""
from __future__ import annotations
import os
import re


LANG_EXTENSIONS = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".toml": "toml",
    ".sh": "shell",
    ".bash": "shell",
}


def list_files(root, ignore_dirs=None, max_files=1000):
    ignore_dirs = set(ignore_dirs or [".git", "node_modules", "__pycache__", ".venv", "venv"])
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for fn in filenames:
            found.append(os.path.join(dirpath, fn))
            if len(found) >= max_files:
                return found
    return found


def language_stats(root, files=None):
    if files is None:
        files = list_files(root)
    stats = {}
    for path in files:
        ext = os.path.splitext(path)[1]
        lang = LANG_EXTENSIONS.get(ext)
        if lang:
            stats[lang] = stats.get(lang, 0) + 1
    return stats


def find_todos(root, files=None, patterns=None):
    if files is None:
        files = list_files(root, max_files=5000)
    patterns = patterns or ["TODO", "FIXME", "XXX", "HACK"]
    matches = []
    for path in files:
        try:
            with open(path, "r", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    for p in patterns:
                        if p in line:
                            matches.append({"path": path, "line": i, "pattern": p, "text": line.strip()})
                            break
        except Exception:
            pass
    return matches


def file_size(path):
    return os.path.getsize(path)


def total_size(root):
    return sum(file_size(p) for p in list_files(root))


def directory_structure(root, max_depth=3, current_depth=0, ignore_dirs=None):
    ignore_dirs = set(ignore_dirs or [".git", "node_modules", "__pycache__"])
    if current_depth > max_depth:
        return {}
    entries = {}
    try:
        for name in sorted(os.listdir(root)):
            if name in ignore_dirs:
                continue
            path = os.path.join(root, name)
            if os.path.isdir(path):
                entries[name] = directory_structure(path, max_depth, current_depth + 1, ignore_dirs)
            else:
                entries[name] = "file"
    except PermissionError:
        pass
    return entries


def main() -> int:
    root = "."
    files = list_files(root)
    stats = language_stats(root, files)
    todos = find_todos(root, files)
    print(f"Files: {len(files)}, Languages: {stats}, TODOs: {len(todos)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())