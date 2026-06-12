"""
Lección: 01-managed-llm-platforms
Fase: 17
Managed LLM platforms: OpenAI, Anthropic,
Google, AWS Bedrock, Azure OpenAI, Together,
Fireworks, Replicate. Comparison of
capabilities, pricing, latency.
"""
from __future__ import annotations


MANAGED_PLATFORMS = {
    "openai": {
        "name": "OpenAI",
        "vendor": "OpenAI",
        "models": ["gpt-4o", "gpt-4o-mini", "o1", "o3-mini"],
        "pricing_per_1m_input": 2.5,
        "pricing_per_1m_output": 10.0,
        "context_window": 128000,
        "year": 2024,
    },
    "anthropic": {
        "name": "Anthropic",
        "vendor": "Anthropic",
        "models": ["claude-3-5-sonnet", "claude-3-5-haiku", "claude-3-opus"],
        "pricing_per_1m_input": 3.0,
        "pricing_per_1m_output": 15.0,
        "context_window": 200000,
        "year": 2024,
    },
    "google": {
        "name": "Google AI",
        "vendor": "Google",
        "models": ["gemini-1.5-pro", "gemini-1.5-flash"],
        "pricing_per_1m_input": 1.25,
        "pricing_per_1m_output": 5.0,
        "context_window": 1000000,
        "year": 2024,
    },
    "bedrock": {
        "name": "AWS Bedrock",
        "vendor": "AWS",
        "models": ["anthropic.*", "amazon.titan", "meta.llama"],
        "pricing_per_1m_input": 0.8,
        "pricing_per_1m_output": 3.2,
        "context_window": 200000,
        "year": 2024,
    },
    "azure": {
        "name": "Azure OpenAI",
        "vendor": "Microsoft",
        "models": ["gpt-4o", "gpt-4-turbo"],
        "pricing_per_1m_input": 2.5,
        "pricing_per_1m_output": 10.0,
        "context_window": 128000,
        "year": 2024,
    },
    "together": {
        "name": "Together AI",
        "vendor": "Together",
        "models": ["llama-3.1-405b", "mixtral-8x22b"],
        "pricing_per_1m_input": 0.9,
        "pricing_per_1m_output": 0.9,
        "context_window": 32000,
        "year": 2024,
    },
    "fireworks": {
        "name": "Fireworks",
        "vendor": "Fireworks",
        "models": ["llama-3.1-405b", "qwen-2.5-72b"],
        "pricing_per_1m_input": 0.7,
        "pricing_per_1m_output": 0.7,
        "context_window": 128000,
        "year": 2024,
    },
}


def list_platforms():
    return list(MANAGED_PLATFORMS.keys())


def get_platform(key):
    return MANAGED_PLATFORMS.get(key)


def cheapest(platforms=None):
    if platforms is None:
        platforms = MANAGED_PLATFORMS
    return min(platforms.items(), key=lambda kv: kv[1]["pricing_per_1m_input"])


def largest_context(platforms=None):
    if platforms is None:
        platforms = MANAGED_PLATFORMS
    return max(platforms.items(), key=lambda kv: kv[1]["context_window"])


def estimate_cost(platform_key, input_tokens, output_tokens):
    p = MANAGED_PLATFORMS.get(platform_key)
    if not p:
        return None
    input_cost = p["pricing_per_1m_input"] * input_tokens / 1_000_000
    output_cost = p["pricing_per_1m_output"] * output_tokens / 1_000_000
    return input_cost + output_cost


def main() -> int:
    print(f"Platforms: {list_platforms()}")
    print(f"Cheapest: {cheapest()[0]}")
    print(f"Cost 1M/100k: ${estimate_cost('openai', 1_000_000, 100_000):.2f}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())