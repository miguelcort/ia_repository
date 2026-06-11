"""
Lección: 14-autogen-actor-model
Fase: 14
AutoGen (Microsoft 2024): actor model para multi-agent
conversation. ConversableAgent, GroupChat, code execution.
+Conversational +Multi-agent.
"""
from __future__ import annotations
from collections import defaultdict


class ConversableAgent:
    """Mock AutoGen ConversableAgent."""
    def __init__(self, name, system_message, llm_config=None, human_input=False):
        self.name = name
        self.system_message = system_message
        self.llm_config = llm_config or {}
        self.human_input = human_input
        self.chat_history = []
        self.reply_count = 0

    def send(self, message, recipient):
        """Send a message to recipient."""
        self.chat_history.append({"role": "user", "content": message})
        return self

    def receive(self, message, sender):
        """Receive a message and generate reply."""
        self.chat_history.append({"role": "user", "content": message})
        # mock reply
        reply = f"{self.name} replying to: {message[:50]}"
        self.chat_history.append({"role": "assistant", "content": reply})
        self.reply_count += 1
        return reply

    def get_history(self):
        return self.chat_history


class GroupChat:
    """Mock AutoGen GroupChat."""
    def __init__(self, agents, max_round=10):
        self.agents = agents
        self.max_round = max_round
        self.messages = []
        self.speaker_history = []

    def run(self, initial_message, manager=None):
        """Run group chat."""
        self.messages.append({"role": "user", "content": initial_message, "name": "User"})
        for r in range(self.max_round):
            # simple round-robin
            agent = self.agents[r % len(self.agents)]
            last = self.messages[-1]["content"]
            reply = agent.receive(last, sender=None)
            self.messages.append({"role": "assistant", "content": reply, "name": agent.name})
            self.speaker_history.append(agent.name)
        return self.messages


class UserProxyAgent(ConversableAgent):
    """Mock AutoGen UserProxyAgent: can execute code."""
    def __init__(self, name="User", code_execution=False, **kwargs):
        super().__init__(name, system_message="User proxy", **kwargs)
        self.code_execution = code_execution

    def execute_code(self, code):
        """Mock code execution."""
        return f"Output: {code[:50]}"


def main() -> int:
    a1 = ConversableAgent("alice", "You are a coder.")
    a2 = ConversableAgent("bob", "You are a reviewer.")
    chat = GroupChat([a1, a2], max_round=4)
    chat.run("Build a calculator")
    print(f"Messages: {len(chat.messages)}")
    print(f"Speakers: {chat.speaker_history}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())