"""
Lección: 19-a2a-protocol
Fase: 13
A2A (Agent-to-Agent, Google 2025): protocolo para comunicacion
agent-to-agent. Agent cards, JSON-RPC, streaming, multi-agent.
"""
from __future__ import annotations
import json


def make_agent_card(name, version, description, capabilities, skills, url):
    """Build agent card (A2A discovery)."""
    return {
        "name": name,
        "version": version,
        "description": description,
        "capabilities": capabilities,
        "skills": skills,
        "url": url,
    }


def make_a2a_message(method, params, msg_id=1):
    """A2A message (JSON-RPC 2.0)."""
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": msg_id,
    }


def a2a_task_send(agent_url, task):
    """tasks/send: send task to agent."""
    return make_a2a_message("tasks/send", {
        "agent": agent_url,
        "task": task,
    })


def a2a_task_get(task_id):
    """tasks/get: get task status."""
    return make_a2a_message("tasks/get", {"id": task_id})


def a2a_task_cancel(task_id):
    """tasks/cancel: cancel task."""
    return make_a2a_message("tasks/cancel", {"id": task_id})


def a2a_stream_subscribe(task_id):
    """streaming/subscribe: SSE updates."""
    return make_a2a_message("streaming/subscribe", {"id": task_id})


def task_payload(prompt, artifacts=None):
    """Build task payload."""
    return {
        "input": {"prompt": prompt},
        "artifacts": artifacts or [],
    }


def main() -> int:
    card = make_agent_card(
        "research-agent", "1.0.0", "Research assistant",
        capabilities=["streaming", "tools"],
        skills=[{"id": "search", "name": "Web search"}],
        url="https://agent.example.com",
    )
    print(json.dumps(card, indent=2))
    msg = a2a_task_send(card["url"], task_payload("What is AI?"))
    print(f"Task send: {msg['method']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())