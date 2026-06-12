"""
Lección: 10-claude-code-permission-modes
Fase: 15
Claude Code permission modes: default,
acceptEdits, plan, dontAsk, bypassPermissions.
"""
from __future__ import annotations


PERMISSION_MODES = {
    "default": {
        "name": "default",
        "description": "Standard behavior: ask for permission on potentially dangerous operations",
        "tools_require_approval": ["bash", "write", "edit"],
        "tools_no_approval": ["read", "search", "grep", "glob"],
    },
    "acceptEdits": {
        "name": "acceptEdits",
        "description": "Auto-accept file edits but still ask for bash commands",
        "tools_require_approval": ["bash"],
        "tools_no_approval": ["read", "search", "grep", "glob", "write", "edit"],
    },
    "plan": {
        "name": "plan",
        "description": "Plan mode: present a plan and wait for user approval before executing",
        "tools_require_approval": ["*"],
        "tools_no_approval": [],
    },
    "dontAsk": {
        "name": "dontAsk",
        "description": "Dont ask mode: deny any tool that requires permission",
        "tools_require_approval": [],
        "tools_no_approval": [],
        "deny_by_default": True,
    },
    "bypassPermissions": {
        "name": "bypassPermissions",
        "description": "Bypass all permission checks (use with caution)",
        "tools_require_approval": [],
        "tools_no_approval": ["*"],
    },
}


def list_permission_modes():
    return list(PERMISSION_MODES.keys())


def get_mode(name):
    return PERMISSION_MODES.get(name)


def requires_approval(mode_name, tool):
    mode = PERMISSION_MODES.get(mode_name)
    if not mode:
        raise ValueError(f"Unknown mode: {mode_name}")
    if mode.get("deny_by_default"):
        return True
    if "*" in mode["tools_no_approval"]:
        return False
    if "*" in mode["tools_require_approval"]:
        return True
    if tool in mode["tools_no_approval"]:
        return False
    if tool in mode["tools_require_approval"]:
        return True
    return True


def plan_step(plan_steps, current):
    """Returns next step or None if done."""
    if current >= len(plan_steps):
        return None
    return plan_steps[current]


def main() -> int:
    print(f"Modes: {list_permission_modes()}")
    print(f"default+bash: {requires_approval('default', 'bash')}")
    print(f"acceptEdits+write: {requires_approval('acceptEdits', 'write')}")
    print(f"bypassPermissions+rm: {requires_approval('bypassPermissions', 'rm')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())