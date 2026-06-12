"""
Lección: 03-communication-protocols
Fase: 16
Communication protocols: direct
messaging, broadcast, pub/sub,
blackboard, request/reply, async
messaging, message passing.
"""
from __future__ import annotations
import time
import uuid


class Message:
    def __init__(self, sender, receiver, content, channel=None, ttl=None):
        self.id = str(uuid.uuid4())
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.channel = channel
        self.ttl = ttl
        self.timestamp = time.time()


class Channel:
    def __init__(self, name):
        self.name = name
        self.subscribers = []
        self.messages = []

    def subscribe(self, agent):
        if agent not in self.subscribers:
            self.subscribers.append(agent)

    def unsubscribe(self, agent):
        if agent in self.subscribers:
            self.subscribers.remove(agent)

    def publish(self, sender, content):
        msg = Message(sender, "broadcast", content, channel=self.name)
        self.messages.append(msg)
        for sub in self.subscribers:
            sub.inbox.append(msg)


class Blackboard:
    def __init__(self):
        self.entries = {}

    def write(self, key, value, author):
        self.entries[key] = {"value": value, "author": author, "ts": time.time()}

    def read(self, key):
        entry = self.entries.get(key)
        return entry["value"] if entry else None

    def keys(self):
        return list(self.entries.keys())


class DirectMailbox:
    def __init__(self, agent_id):
        self.agent_id = agent_id
        self.inbox = []

    def receive(self, msg):
        if msg.receiver == self.agent_id or msg.receiver == "broadcast":
            self.inbox.append(msg)


class PubSub:
    def __init__(self):
        self.channels = {}

    def create_channel(self, name):
        if name not in self.channels:
            self.channels[name] = Channel(name)
        return self.channels[name]

    def get_channel(self, name):
        return self.channels.get(name)


def main() -> int:
    mb = DirectMailbox("a1")
    msg = Message("a2", "a1", "hello")
    mb.receive(msg)
    print(f"Inbox: {len(mb.inbox)}")
    bb = Blackboard()
    bb.write("task", "refactor", "a1")
    print(f"Blackboard: {bb.read('task')}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())