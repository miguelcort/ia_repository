"""
Lección: 04-primitive-model
Fase: 16
Primitive model: minimal agent
with message handlers, state, tick,
react to messages, send messages.
"""
from __future__ import annotations
import time
import uuid


class PrimitiveAgent:
    def __init__(self, agent_id=None, name=None):
        self.agent_id = agent_id or str(uuid.uuid4())
        self.name = name or self.agent_id
        self.inbox = []
        self.outbox = []
        self.state = {}
        self.handlers = {}

    def on(self, performative, fn):
        self.handlers[performative] = fn

    def receive(self, message):
        self.inbox.append(message)

    def _dispatch(self, message):
        handler = self.handlers.get(message.performative)
        if handler:
            result = handler(message)
            if result is not None:
                self.send(message.sender, "inform", result)

    def send(self, receiver, performative, content):
        msg = Message(self.agent_id, receiver, content, performative=performative)
        self.outbox.append(msg)
        return msg

    def tick(self):
        """Process one step: handle all pending messages."""
        while self.inbox:
            msg = self.inbox.pop(0)
            self._dispatch(msg)

    def set_state(self, key, value):
        self.state[key] = value

    def get_state(self, key, default=None):
        return self.state.get(key, default)


class Message:
    def __init__(self, sender, receiver, content, performative="inform",
                 channel=None, ttl=None):
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.performative = performative
        self.channel = channel
        self.ttl = ttl
        self.timestamp = time.time()


def main() -> int:
    a = PrimitiveAgent(name="alice")
    a.on("ping", lambda m: "pong")
    a.receive(Message("bob", a.agent_id, "hi", performative="ping"))
    a.tick()
    print(f"Outbox: {len(a.outbox)}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())