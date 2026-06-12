"""
Lección: 13-shared-memory-blackboard
Fase: 16
Shared memory / blackboard: shared
state for multi-agent, key-value with
locks, version, access log, pattern
match queries.
"""
from __future__ import annotations
import time


class Blackboard:
    def __init__(self):
        self._data = {}
        self._version = {}
        self._log = []
        self._locks = set()

    def write(self, key, value, author):
        if key in self._locks:
            raise RuntimeError(f"key {key} is locked")
        self._data[key] = value
        self._version[key] = self._version.get(key, 0) + 1
        self._log.append({"op": "write", "key": key, "author": author, "ts": time.time()})

    def read(self, key):
        return self._data.get(key)

    def version(self, key):
        return self._version.get(key, 0)

    def lock(self, key, author):
        if key in self._locks:
            raise RuntimeError(f"key {key} already locked")
        self._locks.add(key)
        self._log.append({"op": "lock", "key": key, "author": author, "ts": time.time()})

    def unlock(self, key, author):
        if key in self._locks:
            self._locks.remove(key)
            self._log.append({"op": "unlock", "key": key, "author": author, "ts": time.time()})

    def is_locked(self, key):
        return key in self._locks

    def pattern_query(self, pattern):
        import re
        rx = re.compile(pattern)
        return {k: v for k, v in self._data.items() if rx.search(k)}

    def access_log(self, key=None):
        if key is None:
            return list(self._log)
        return [e for e in self._log if e["key"] == key]

    def keys(self):
        return list(self._data.keys())


def main() -> int:
    bb = Blackboard()
    bb.write("task", "refactor", "a1")
    bb.write("language", "python", "a1")
    bb.lock("task", "a2")
    print(f"Locked: {bb.is_locked('task')}")
    print(f"Pattern: {bb.pattern_query('.*a.*')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())