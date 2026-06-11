"""
Lección: 13-mcp-async-tasks
Fase: 13
MCP async tasks: long-running operations. task/create, task/get,
task/list, task/cancel. Polling vs streaming. Task status.
"""
from __future__ import annotations
import time
import json


TASK_STATUS_PENDING = "pending"
TASK_STATUS_RUNNING = "running"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"
TASK_STATUS_CANCELLED = "cancelled"


class Task:
    """Mock async task."""
    def __init__(self, task_id, operation, params=None):
        self.id = task_id
        self.operation = operation
        self.params = params or {}
        self.status = TASK_STATUS_PENDING
        self.result = None
        self.error = None
        self.created_at = time.time()
        self.completed_at = None

    def to_dict(self):
        return {
            "id": self.id,
            "operation": self.operation,
            "status": self.status,
            "result": self.result,
            "error": self.error,
            "createdAt": self.created_at,
            "completedAt": self.completed_at,
        }


class TaskManager:
    """Mock MCP task manager."""
    def __init__(self):
        self.tasks = {}
        self._next_id = 1

    def create_task(self, operation, params=None):
        """task/create."""
        task_id = f"task_{self._next_id}"
        self._next_id += 1
        task = Task(task_id, operation, params)
        self.tasks[task_id] = task
        return task

    def get_task(self, task_id):
        """task/get."""
        return self.tasks.get(task_id)

    def list_tasks(self, status=None):
        """task/list."""
        if status:
            return [t for t in self.tasks.values() if t.status == status]
        return list(self.tasks.values())

    def cancel_task(self, task_id):
        """task/cancel."""
        task = self.tasks.get(task_id)
        if not task:
            return None
        if task.status in (TASK_STATUS_COMPLETED, TASK_STATUS_FAILED, TASK_STATUS_CANCELLED):
            return False
        task.status = TASK_STATUS_CANCELLED
        task.completed_at = time.time()
        return True

    def mark_running(self, task_id):
        task = self.tasks.get(task_id)
        if task:
            task.status = TASK_STATUS_RUNNING

    def mark_completed(self, task_id, result):
        task = self.tasks.get(task_id)
        if task:
            task.status = TASK_STATUS_COMPLETED
            task.result = result
            task.completed_at = time.time()

    def mark_failed(self, task_id, error):
        task = self.tasks.get(task_id)
        if task:
            task.status = TASK_STATUS_FAILED
            task.error = error
            task.completed_at = time.time()


def main() -> int:
    tm = TaskManager()
    t = tm.create_task("long_op", {"n": 10})
    print(f"Created: {t.id}, status: {t.status}")
    tm.mark_running(t.id)
    tm.mark_completed(t.id, {"output": 42})
    print(f"After: {t.status}, result: {t.result}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())