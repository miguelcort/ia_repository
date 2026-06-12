"""
Lección: 34-repo-memory-and-state
Fase: 14
Repo memory + state: persistent context
across sessions, file-based state,
git-tracked memory, structured
storage + retrieval.
"""
from __future__ import annotations
import json
import os
import time
import uuid


class RepoMemory:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)
        self.index_path = os.path.join(self.base_dir, "index.json")
        self.entries = {}
        self._load()

    def _load(self):
        if os.path.exists(self.index_path):
            with open(self.index_path) as f:
                self.entries = json.load(f)

    def _save(self):
        with open(self.index_path, "w") as f:
            json.dump(self.entries, f, indent=2)

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
        entry["_path"] = path
        self.entries[key] = entry
        self._save()
        return entry

    def get(self, key):
        entry = self.entries.get(key)
        if not entry:
            return None
        if "expires_at" in entry and time.time() > entry["expires_at"]:
            del self.entries[key]
            self._save()
            return None
        return entry

    def search(self, query, tag=None):
        results = []
        for key, entry in self.entries.items():
            if tag and tag not in entry.get("tags", []):
                continue
            if query.lower() in key.lower() or query.lower() in str(entry.get("value", "")).lower():
                results.append(entry)
        return results

    def delete(self, key):
        if key in self.entries:
            entry = self.entries.pop(key)
            path = entry.get("_path")
            if path and os.path.exists(path):
                os.remove(path)
            self._save()
            return True
        return False

    def list_keys(self):
        return list(self.entries.keys())


def main() -> int:
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        m = RepoMemory(tmp)
        m.add("project", "agent workbench", tags=["meta"])
        m.add("language", "python", tags=["meta"])
        print(m.get("project"))
        print(f"Search: {len(m.search('python'))}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())