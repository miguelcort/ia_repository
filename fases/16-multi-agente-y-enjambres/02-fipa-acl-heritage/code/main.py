"""
Lección: 02-fipa-acl-heritage
Fase: 16
FIPA ACL heritage: Foundation for
Intelligent Physical Agents, Agent
Communication Language, performatives,
message structure, ontology.
"""
from __future__ import annotations
import time
import uuid


PERFORMATIVES = {
    "inform": {
        "name": "inform",
        "description": "Tell another agent a fact",
        "sender": "informant",
        "receiver": "informed",
        "example": "The price is 100",
    },
    "request": {
        "name": "request",
        "description": "Ask another agent to perform an action",
        "sender": "requester",
        "receiver": "performer",
        "example": "Please compute X",
    },
    "query": {
        "name": "query",
        "description": "Ask for information",
        "sender": "asker",
        "receiver": "answerer",
        "example": "What is the status?",
    },
    "propose": {
        "name": "propose",
        "description": "Suggest a course of action",
        "sender": "proposer",
        "receiver": "decider",
        "example": "I propose we use algorithm A",
    },
    "accept": {
        "name": "accept-proposal",
        "description": "Accept a proposal",
        "sender": "decider",
        "receiver": "proposer",
        "example": "I accept your proposal",
    },
    "reject": {
        "name": "reject-proposal",
        "description": "Reject a proposal",
        "sender": "decider",
        "receiver": "proposer",
        "example": "I reject that proposal",
    },
    "cfp": {
        "name": "call-for-proposal",
        "description": "Request proposals from agents",
        "sender": "manager",
        "receiver": "contractors",
        "example": "CFP for task X",
    },
}


def list_performatives():
    return list(PERFORMATIVES.keys())


def get_performative(name):
    return PERFORMATIVES.get(name)


class ACLMessage:
    def __init__(self, performative, sender, receiver, content,
                 ontology=None, conversation_id=None, reply_with=None):
        self.performative = performative
        self.sender = sender
        self.receiver = receiver
        self.content = content
        self.ontology = ontology
        self.conversation_id = conversation_id or str(uuid.uuid4())
        self.reply_with = reply_with or str(uuid.uuid4())
        self.timestamp = time.time()

    def to_dict(self):
        return {
            "performative": self.performative,
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "ontology": self.ontology,
            "conversation_id": self.conversation_id,
            "reply_with": self.reply_with,
            "timestamp": self.timestamp,
        }


def main() -> int:
    print(f"Performatives: {list_performatives()}")
    msg = ACLMessage("inform", "a1", "a2", "hello")
    print(msg.to_dict())
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())