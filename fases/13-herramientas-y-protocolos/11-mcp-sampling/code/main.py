"""
Lección: 11-mcp-sampling
Fase: 13
MCP sampling: server asks client to run LLM. Recursive LLM use.
sampling/createMessage. Human-in-the-loop. Tokens, model prefs.
"""
from __future__ import annotations
import json


def make_sampling_request(messages, model_prefs=None, max_tokens=1024,
                          system_prompt=None, temperature=0.7, include_context="none"):
    """Build sampling/createMessage request."""
    params = {
        "messages": messages,
        "maxTokens": max_tokens,
        "temperature": temperature,
        "includeContext": include_context,
    }
    if model_prefs:
        params["modelPreferences"] = model_prefs
    if system_prompt:
        params["systemPrompt"] = system_prompt
    return {"jsonrpc": "2.0", "method": "sampling/createMessage", "params": params, "id": 1}


def make_sampling_response(text, model="claude-3-5-sonnet", stop_reason="endTurn"):
    """Build sampling/createMessage response."""
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "result": {
            "model": model,
            "stopReason": stop_reason,
            "role": "assistant",
            "content": {"type": "text", "text": text},
        },
    }


def model_preferences(cost_priority=0.5, speed_priority=0.5, intelligence_priority=0.5,
                      hints=None):
    """Model preferences con priorities."""
    return {
        "costPriority": cost_priority,
        "speedPriority": speed_priority,
        "intelligencePriority": intelligence_priority,
        "hints": hints or [],
    }


def include_context_options():
    """Available includeContext values."""
    return ["none", "thisServer", "allServers"]


def sampling_with_human_approval(client, server_request):
    """Mock sampling con human-in-the-loop approval."""
    # In real: prompt user antes de ejecutar
    return {"approved": True, "result": {"text": "mock response"}}


def main() -> int:
    req = make_sampling_request(
        messages=[{"role": "user", "content": {"type": "text", "text": "Hello"}}],
        model_prefs=model_preferences(intelligence_priority=0.8),
        max_tokens=100,
    )
    print(json.dumps(req, indent=2))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())