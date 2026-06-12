"""
Lección: 19-ai-gateways
Fase: 17
AI gateways: Portkey, LiteLLM, OpenRouter,
unified API across providers,
routing, fallbacks, caching, rate
limits, observability.
"""
from __future__ import annotations


class Provider:
    def __init__(self, name, base_url, api_key=None, models=None):
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.models = models or []


class AIGateway:
    def __init__(self):
        self.providers = {}
        self.routes = {}
        self.cache = {}

    def add_provider(self, provider):
        self.providers[provider.name] = provider

    def add_route(self, alias, providers):
        """Map an alias to an ordered list of providers (for fallback)."""
        self.routes[alias] = providers

    def call(self, alias, model, prompt):
        cache_key = f"{alias}:{model}:{prompt}"
        if cache_key in self.cache:
            return self.cache[cache_key], "cache"
        providers = self.routes.get(alias, [])
        for provider_name in providers:
            provider = self.providers.get(provider_name)
            if provider is None:
                continue
            if model in provider.models or not provider.models:
                response = f"[{provider.name}] response to: {prompt[:30]}"
                self.cache[cache_key] = response
                return response, provider.name
        return None, "no_provider"

    def list_routes(self):
        return list(self.routes.keys())


def main() -> int:
    gw = AIGateway()
    gw.add_provider(Provider("openai", "https://api.openai.com", models=["gpt-4o"]))
    gw.add_provider(Provider("anthropic", "https://api.anthropic.com", models=["claude-3-5-sonnet"]))
    gw.add_route("smart", ["anthropic", "openai"])
    print(gw.call("smart", "claude-3-5-sonnet", "Hello"))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())