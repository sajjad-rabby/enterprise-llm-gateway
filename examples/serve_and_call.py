"""Start the gateway server, call it over HTTP, then read the metrics.

Demonstrates that any OpenAI-compatible client can point at the gateway
without knowing which provider actually served the request.

Usage::

    python examples/serve_and_call.py
"""

from __future__ import annotations

import json
import os
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from enterprise_llm_gateway.models import ModelResponse, Usage  # noqa: E402
from enterprise_llm_gateway.server import GatewayServer  # noqa: E402
from demo_end_to_end import build_demo_gateway  # noqa: E402


def main() -> int:
    """Boot the server, issue one chat call, and scrape /metrics."""
    gateway = build_demo_gateway()
    server = GatewayServer(gateway, host="127.0.0.1", port=0)
    host, port = server.start_background()
    base = f"http://{host}:{port}"
    print(f"gateway listening on {base}\n")

    try:
        # 1. Liveness probe
        with urllib.request.urlopen(f"{base}/healthz", timeout=5) as response:
            health = json.loads(response.read().decode("utf-8"))
        print(f"GET /healthz  -> {response.status}  {health}")

        # 2. Which models can the gateway route to?
        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as response:
            models = json.loads(response.read().decode("utf-8"))
        print("\nGET /v1/models")
        for entry in models["data"]:
            print(f"  {entry['id']:<28} via {entry['provider']}")

        # 3. An OpenAI-shaped chat completion request
        payload = json.dumps({
            "model": "auto",
            "messages": [{"role": "system", "content": "You are terse."},
                         {"role": "user", "content": "What is a circuit breaker?"}],
            "max_tokens": 64,
            "user": "example-client",
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{base}/v1/chat/completions", data=payload, method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))

        print("\nPOST /v1/chat/completions")
        print(f"  id            : {result['id']}")
        print(f"  served by     : {result['provider']}")
        print(f"  content       : {result['choices'][0]['message']['content']}")
        print(f"  usage         : {result['usage']}")
        print(f"  gateway meta  : {result['x_gateway']}")

        # 4. Repeat the call to show the cache absorbing it
        with urllib.request.urlopen(request, timeout=5) as response:
            repeat = json.loads(response.read().decode("utf-8"))
        print(f"\n  repeat call cache_hit = {repeat['x_gateway']['cache_hit']}")

        # 5. Prometheus scrape
        with urllib.request.urlopen(f"{base}/metrics", timeout=5) as response:
            metrics = response.read().decode("utf-8")
        print("\nGET /metrics (excerpt)")
        for line in metrics.splitlines():
            if line.startswith("gateway_cache_hit_rate") or \
               line.startswith("gateway_total_cost_usd") or \
               line.startswith("gateway_requests_total"):
                print(f"  {line}")

    finally:
        server.shutdown()
        print("\nserver shut down cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
