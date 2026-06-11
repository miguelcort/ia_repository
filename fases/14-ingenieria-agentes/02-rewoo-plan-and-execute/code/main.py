"""
Lección: 02-rewoo-plan-and-execute
Fase: 14
ReWoo (Reasoning Without Observation): plan-and-execute.
Planner genera DAG de tasks, executor ejecuta paralelo.
+Token-efficient que ReAct. Xu et al. 2023.
"""
from __future__ import annotations
import re
from collections import defaultdict


def parse_rewoo_plan(plan_text):
    """Parse ReWoo plan: #E1 = tool(args), #E2 depends on #E1."""
    tasks = []
    pattern = r"#E(\d+)\s*=\s*(\w+)\[(.+?)\](?:\s*\(.*?\))?"
    for match in re.finditer(pattern, plan_text, re.DOTALL):
        task_id = int(match.group(1))
        tool = match.group(2)
        args_text = match.group(3)
        tasks.append({
            "id": task_id,
            "tool": tool,
            "args": args_text.strip(),
            "deps": [],  # populated below
        })
    # detect dependencies (#E1, #E2 in args)
    for task in tasks:
        deps = [int(m) for m in re.findall(r"#E(\d+)", task["args"])]
        task["deps"] = deps
    return tasks


def build_dag(tasks):
    """Build DAG adjacency from tasks."""
    dag = defaultdict(list)
    for task in tasks:
        for dep in task["deps"]:
            dag[dep].append(task["id"])
    return dict(dag)


def topological_order(tasks):
    """Topological order of tasks."""
    visited = set()
    order = []
    def visit(task_id):
        if task_id in visited:
            return
        visited.add(task_id)
        task = next(t for t in tasks if t["id"] == task_id)
        for dep in task["deps"]:
            visit(dep)
        order.append(task_id)
    for task in tasks:
        visit(task["id"])
    return order


def execute_plan(tasks, tool_registry, max_depth=10):
    """Execute ReWoo plan: topological order."""
    results = {}
    order = topological_order(tasks)
    for task_id in order:
        task = next(t for t in tasks if t["id"] == task_id)
        # resolve args (replace #E1 with results)
        args = task["args"]
        for dep_id in task["deps"]:
            args = args.replace(f"#E{dep_id}", str(results[dep_id]))
        tool_fn = tool_registry.get(task["tool"])
        if tool_fn:
            try:
                results[task_id] = tool_fn(args)
            except Exception as e:
                results[task_id] = f"error: {e}"
        else:
            results[task_id] = f"unknown tool: {task['tool']}"
    return results


def main() -> int:
    plan = """
    #E1 = LLM[What is the capital of France?]
    #E2 = Search[#E1 + " population"]
    #E3 = LLM[Summarize: #E1 and #E2]
    """
    tasks = parse_rewoo_plan(plan)
    print(f"Tasks: {len(tasks)}")
    dag = build_dag(tasks)
    print(f"DAG: {dag}")
    order = topological_order(tasks)
    print(f"Order: {order}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())