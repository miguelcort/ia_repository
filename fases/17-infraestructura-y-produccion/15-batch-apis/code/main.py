"""
Lección: 15-batch-apis
Fase: 15
Batch APIs: OpenAI/Anthropic batch
endpoints, 50% cost discount, async
processing, results in 24h, JSONL input,
use cases: bulk eval, dataset processing.
"""
from __future__ import annotations
import json
import time


class BatchRequest:
    def __init__(self, custom_id, method, url, body):
        self.custom_id = custom_id
        self.method = method
        self.url = url
        self.body = body


class BatchJob:
    def __init__(self, requests=None):
        self.requests = list(requests or [])
        self.created_at = time.time()
        self.completed_at = None
        self.results = []

    def add(self, request):
        self.requests.append(request)

    def to_jsonl(self):
        lines = []
        for r in self.requests:
            lines.append(json.dumps({
                "custom_id": r.custom_id,
                "method": r.method,
                "url": r.url,
                "body": r.body,
            }))
        return "\n".join(lines)

    def total(self):
        return len(self.requests)

    def estimated_cost(self, per_request=0.005, discount=0.5):
        return self.total() * per_request * (1 - discount)

    def mark_complete(self, results):
        self.completed_at = time.time()
        self.results = results


def parse_jsonl_results(jsonl_text):
    results = []
    for line in jsonl_text.strip().split("\n"):
        if not line:
            continue
        results.append(json.loads(line))
    return results


def main() -> int:
    job = BatchJob()
    job.add(BatchRequest("r1", "POST", "/v1/chat/completions", {"model": "gpt-4o-mini", "messages": []}))
    print(f"Total: {job.total()}, est cost: ${job.estimated_cost():.4f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())