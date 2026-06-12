"""
Lección: 17-disaggregated-prefill-decode
Fase: 17
Disaggregated prefill-decode: separate
GPUs for prefill (compute-bound) and
decode (memory-bound), Mooncake,
DistServe, throughput, latency.
"""
from __future__ import annotations
import time


class PrefillEngine:
    def __init__(self, name, compute_tflops):
        self.name = name
        self.compute_tflops = compute_tflops
        self.queue = []
        self.busy = False

    def prefill(self, prompt_tokens):
        if self.busy:
            return None
        self.busy = True
        time.sleep(0.001)
        kv = {"prompt_len": len(prompt_tokens), "tokens": prompt_tokens[:]}
        self.busy = False
        return kv


class DecodeEngine:
    def __init__(self, name, memory_gb):
        self.name = name
        self.memory_gb = memory_gb
        self.active = []

    def decode_step(self, kv):
        return {"next_token": "x", "kv_state": kv}


class DisaggregatedEngine:
    def __init__(self, prefill, decode):
        self.prefill = prefill
        self.decode = decode

    def generate(self, prompt_tokens, max_tokens=10):
        kv = self.prefill.prefill(prompt_tokens)
        if kv is None:
            return None
        results = []
        for _ in range(max_tokens):
            step = self.decode.decode_step(kv)
            results.append(step["next_token"])
            kv = step["kv_state"]
        return results


def main() -> int:
    prefill = PrefillEngine("A100", 312)
    decode = DecodeEngine("H100", 80)
    engine = DisaggregatedEngine(prefill, decode)
    print(engine.generate("hello world", max_tokens=5))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())