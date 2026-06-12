"""
Lección: 12-a2a-protocol
Fase: 16
A2A protocol: Agent-to-Agent protocol
by Google, agent cards, JSON-RPC
messaging, interop standard, modern
alternative to FIPA ACL.
"""
from __future__ import annotations
import uuid


A2A_VERSIONS = ["0.1", "0.2", "0.3"]


def create_agent_card(name, description, skills=None, url=None, version="0.3"):
    """Build an A2A-compatible agent card."""
    return {
        "name": name,
        "description": description,
        "url": url or f"https://agents.example.com/{name}",
        "version": version,
        "skills": skills or [],
        "capabilities": {
            "streaming": True,
            "pushNotifications": False,
        },
    }


def add_skill(card, name, description, examples=None):
    card["skills"].append({
        "id": str(uuid.uuid4()),
        "name": name,
        "description": description,
        "examples": examples or [],
    })
    return card


def a2a_message(sender, receiver, parts, message_id=None, role="agent"):
    """Build an A2A JSON-RPC message."""
    return {
        "jsonrpc": "2.0",
        "id": message_id or str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "role": role,
                "parts": parts,
                "from": sender,
                "to": receiver,
            }
        },
    }


def a2a_text_part(text):
    return {"type": "text", "text": text}


def a2a_data_part(data):
    return {"type": "data", "data": data}


def parse_a2a_message(msg):
    """Extract sender, receiver, parts from an A2A message."""
    params = msg.get("params", {})
    inner = params.get("message", {})
    return {
        "sender": inner.get("from"),
        "receiver": inner.get("to"),
        "role": inner.get("role"),
        "parts": inner.get("parts", []),
    }


def main() -> int:
    card = create_agent_card("researcher", "Research agent", skills=["search", "summarize"])
    print(f"Card: {card['name']}")
    msg = a2a_message("a1", "a2", [a2a_text_part("hello")])
    print(f"Message: {parse_a2a_message(msg)['sender']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())