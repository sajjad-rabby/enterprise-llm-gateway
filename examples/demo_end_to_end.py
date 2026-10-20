"""End-to-end demo: caching, routing, failover, and cost attribution.

Runs entirely offline against a fake provider so the output is
deterministic and safe to run in CI or a demo environment.

Usage::

    python examples/demo_end_to_end.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enterprise_llm_gateway import Gateway, ModelRequest  # noqa: E402
from enterprise_llm_gateway.gateway import Gateway as _Gateway  # noqa: E402
from enterprise_llm_gateway.models import ModelRequest as _Request  # noqa: E402
from enterprise_llm_gateway.providers.base import Provider, ProviderError  # noqa: E402
from enterprise_llm_gateway.models import ModelResponse, Usage  # noqa: E402


class DemoProvider(Provider):
    """A deterministic stand-in for a real provider."""

    def __init__(self, name: str, config: dict) -> None:
        super().__init__(name, config)
        self.label = config.get("label", name)
        self.calls = 0
        self.failing = config.get("failing", False)
        self.price_in = float(config.get("price_per_1k_input", 0.0002))
        self.price_out = float(config.get("price_per_1k_output", 0.0006))

    def complete(self, request):
        self.calls += 1
        if self.failing:
            raise ProviderError(self.name, "simulated upstream 503", status_code=503)
        return ModelResponse(
            content=f"[{self.label}] handled {len(request.prompt)} chars",
            model=self.default_model or "demo-model",
            provider=self.name,
            usage=Usage(12, 8),
        )

    def estimate_cost(self, tokens_in: int, tokens_out: int, model=None) -> float:
        return self._cost_from_prices(tokens_in, tokens_out,
                                      self.price_in, self.price_out)


def build_demo_gateway() -> _Gateway:
    """Build a gateway wired to in-process demo providers."""
    providers = {
        "primary": DemoProvider("primary", {
            "default_model": "gpt-4o-mini", "label": "PRIMARY"}),
        "expensive": DemoProvider("expensive", {
            "default_model": "gpt-4o", "label": "EXPENSIVE",
            "price_per_1k_input": 0.0025, "price_per_1k_output": 0.01}),
        "sick": DemoProvider("sick", {
            "default_model": "broken", "failing": True}),
    }
    config = {
        "routing": {"strategy": "cheapest",
                    "candidates": ["primary", "expensive", "sick"],
                    "max_attempts": 3, "fail_threshold": 2},
        "cache": {"enabled": True, "ttl_seconds": 300, "max_entries": 100},
        "ledger": {},
        "providers": {},
    }
    gateway = _Gateway.__new__(_Gateway)
    gateway.config = config
    gateway.providers = providers

    from enterprise_llm_gateway.cache import ResponseCache
    from enterprise_llm_gateway.ledger import UsageLedger
    from enterprise_llm_gateway.router import Router

    gateway.cache = ResponseCache(ttl_seconds=300, max_entries=100)
    gateway.ledger = UsageLedger(path=None)
    gateway.router = Router(providers, config["routing"])
    return gateway


def main() -> int:
    """Run the demo scenarios and print a report."""
    gateway = build_demo_gateway()

    print("=" * 66)
    print("1. CHEAPEST ROUTING - the cost-optimal provider should win")
    print("=" * 66)
    result = gateway.complete(_Request(prompt="Summarise this week's incidents."))
    print(f"  served by   : {result.provider}")
    print(f"  content     : {result.response.content}")
    print(f"  tokens      : {result.response.usage.to_dict()}")
    print(f"  latency     : {result.total_latency_ms:.2f} ms")
    print(f"  candidates  : {gateway.router.candidates}")

    print()
    print("=" * 66)
    print("2. CACHE HIT - the identical prompt never reaches a provider")
    print("=" * 66)
    before = gateway.providers["primary"].calls
    cached = gateway.complete(_Request(prompt="Summarise this week's incidents."))
    after = gateway.providers["primary"].calls
    print(f"  served by   : {cached.provider} (cached={cached.cached})")
    print(f"  provider calls during this step: {after - before}")
    print(f"  cache hit rate: {gateway.cache.stats()['hit_rate']:.2f}")

    print()
    print("=" * 66)
    print("3. FAILOVER - a sick provider is tried, fails, and is skipped")
    print("=" * 66)
    # Put the failing provider first so the request genuinely has to fail
    # over to a healthy one.
    gateway.router.candidates = ["sick", "primary"]
    gateway.router.strategy = "priority"
    for attempt in range(1, 4):
        # A unique prompt each time, otherwise the cache would answer and
        # the routing path would never be exercised.
        result = gateway.complete(_Request(prompt=f"force failover #{attempt}"))
        health = {h["provider"]: h for h in gateway.router.all_health()}
        sick = health.get("sick", {})
        print(f"  attempt {attempt}: served by {result.provider:<8} "
              f"via {result.attempts} "
              f"| sick consecutive failures={sick.get('consecutive_failures')}")
    print(f"  order once degraded: {gateway.router.order(_Request(prompt='x'))}")
    gateway.router.candidates = ["primary", "expensive", "sick"]
    gateway.router.strategy = "cheapest"

    print()
    print("=" * 66)
    print("4. COST ATTRIBUTION - the ledger answers 'where did spend go?'")
    print("=" * 66)
    summary = gateway.ledger.summary()
    print(json.dumps(summary, indent=2))

    print()
    print("=" * 66)
    print("5. OPERATIONAL SNAPSHOT")
    print("=" * 66)
    stats = gateway.stats()
    print(f"  providers      : {', '.join(stats['providers'])}")
    print(f"  strategy       : {stats['routing']['strategy']}")
    print(f"  cache entries  : {stats['cache']['entries']}")
    print(f"  cache hit rate : {stats['cache']['hit_rate']:.2f}")
    print(f"  total spend    : ${stats['ledger']['total_cost_usd']:.6f}")
    print(f"  p95 latency    : {stats['ledger']['p95_latency_ms']:.2f} ms")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
