"""
Lección: 21-computer-use-agents
Fase: 14
Computer use agents: AI control mouse + keyboard + screen.
Anthropic Claude, OpenAI Operator, OS-Atlas, ShowUI, UI-TARS.
+Production computer use.
"""
from __future__ import annotations
import time


class Screen:
    """Mock screen."""
    def __init__(self, width=1920, height=1080):
        self.width = width
        self.height = height
        self.elements = []  # list of (x, y, w, h, type, label)

    def add_element(self, x, y, w, h, type, label):
        self.elements.append({"x": x, "y": y, "w": w, "h": h, "type": type, "label": label})

    def find_element(self, label):
        for el in self.elements:
            if label.lower() in el["label"].lower():
                cx = el["x"] + el["w"] // 2
                cy = el["y"] + el["h"] // 2
                return (cx, cy)
        return None

    def screenshot(self):
        return "mock_screenshot"


class ComputerUseAgent:
    """Mock computer use agent."""
    def __init__(self, name, model="claude-3-5-sonnet"):
        self.name = name
        self.model = model
        self.actions = []
        self.history = []

    def observe(self, screen):
        """Observe screen."""
        return screen.screenshot()

    def predict_action(self, observation, instruction):
        """Predict next action (click, type, scroll)."""
        # mock: parse simple instruction
        if "click" in instruction.lower():
            return {"action": "click", "target": instruction.split("click")[-1].strip()}
        elif "type" in instruction.lower():
            return {"action": "type", "text": instruction.split("type")[-1].strip()}
        else:
            return {"action": "done"}

    def execute(self, action, screen):
        """Execute action on screen."""
        self.actions.append(action)
        if action["action"] == "click":
            coords = screen.find_element(action["target"])
            if coords:
                return f"clicked at {coords}"
            return f"element not found: {action['target']}"
        elif action["action"] == "type":
            return f"typed: {action['text']}"
        return "done"

    def run(self, screen, instruction, max_steps=10):
        """Run agent loop."""
        for step in range(max_steps):
            obs = self.observe(screen)
            action = self.predict_action(obs, instruction)
            result = self.execute(action, screen)
            self.history.append({"step": step, "action": action, "result": result})
            if action["action"] == "done":
                return self.history
        return self.history


def main() -> int:
    screen = Screen()
    screen.add_element(100, 100, 200, 50, "button", "Login")
    agent = ComputerUseAgent("claude-cu")
    history = agent.run(screen, "click Login")
    print(f"Actions: {len(agent.actions)}")
    print(f"First: {history[0]}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())