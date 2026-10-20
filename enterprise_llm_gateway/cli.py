"""Command-line interface for the gateway.

Subcommands:
- ``providers`` list configured providers and their default models

``serve``     start the OpenAI-compatible HTTP server
``complete``  send a single prompt through the gateway and print the reply
``validate``  check a configuration file without contacting any provider
``stats``     print cache, routing, and spend statistics
``providers`` list configured providers and their default models
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence

from . import __version__
from .config import ConfigError, load_config
from .gateway import Gateway
from .models import ModelRequest
from .router import RoutingError
from .server import GatewayServer


def build_parser() -> argparse.ArgumentParser:
    """Construct the full argument parser for every subcommand."""
    parser = argparse.ArgumentParser(
        prog="llm-gateway",
        description="Multi-provider LLM gateway with caching, routing, and cost tracking.",
    )
    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    serve = subparsers.add_parser("serve", help="run the HTTP server")
    serve.add_argument("--config", required=True, help="path to gateway config JSON")
    serve.add_argument("--host", default="127.0.0.1", help="bind address")
    serve.add_argument("--port", type=int, default=8080, help="bind port")
    serve.add_argument("--verbose", action="store_true", help="log each request")

    complete = subparsers.add_parser("complete", help="send a single prompt")
    complete.add_argument("--config", required=True, help="path to gateway config JSON")
    complete.add_argument("prompt", help="the prompt text to send")
    complete.add_argument("--model", help="model override")
    complete.add_argument("--temperature", type=float, help="sampling temperature")
    complete.add_argument("--max-tokens", type=int, help="completion token cap")
    complete.add_argument("--user", help="caller identifier for cost attribution")
    complete.add_argument("--json", action="store_true",
                          help="print the full OpenAI-shaped response")
    complete.add_argument("--no-cache", action="store_true",
                          help="bypass the response cache for this call")

    validate = subparsers.add_parser("validate", help="validate a config file")
    validate.add_argument("--config", required=True, help="path to gateway config JSON")

    stats = subparsers.add_parser("stats", help="print gateway statistics")
    stats.add_argument("--config", required=True, help="path to gateway config JSON")

    providers = subparsers.add_parser("providers", help="list configured providers")
    providers.add_argument("--config", required=True, help="path to gateway config JSON")

    return parser


def _load(config_path: str) -> dict:
    """Load a configuration file, exiting with a clear message on failure."""
    try:
        return load_config(config_path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)


def _cmd_serve(args: argparse.Namespace) -> int:
    """Run the blocking HTTP server."""
    gateway = Gateway(_load(args.config))
    server = GatewayServer(gateway, host=args.host, port=args.port, verbose=args.verbose)
    host, port = server.address
    print(f"enterprise-llm-gateway {__version__} listening on http://{host}:{port}")
    print(f"  strategy : {gateway.router.strategy}")
    print(f"  providers: {', '.join(sorted(gateway.providers)) or 'none'}")
    print("  endpoints: POST /v1/chat/completions  GET /v1/models  GET /metrics  GET /healthz")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    return 0


def _cmd_complete(args: argparse.Namespace) -> int:
    """Send one prompt through the gateway."""
    gateway = Gateway(_load(args.config))
    if args.no_cache:
        gateway.cache.enabled = False
    request = ModelRequest(
        prompt=args.prompt,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        user=args.user,
    )
    try:
        result = gateway.complete(request)
    except RoutingError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if exc.attempts:
            print("attempted: " + ", ".join(exc.attempts), file=sys.stderr)
        return 1

    if args.json:
        payload = result.response.to_openai_json()
        payload["x_gateway"] = result.to_dict()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(result.response.content)
        source = "cache" if result.cached else result.provider
        usage = result.response.usage
        print(f"\n-- served by {source} | model {result.response.model} | "
              f"{usage.total_tokens} tokens | {result.total_latency_ms:.0f} ms",
              file=sys.stderr)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    """Validate a configuration file and print a summary."""
    config = _load(args.config)
    providers = config.get("providers", {})
    routing = config.get("routing", {})
    print(f"config OK: {args.config}")
    print(f"  providers        : {len(providers)} ({', '.join(sorted(providers))})")
    print(f"  routing strategy : {routing.get('strategy')}")
    print(f"  candidates       : {', '.join(routing.get('candidates', []))}")
    print(f"  max attempts     : {routing.get('max_attempts', 3)}")
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    """Print cache, routing, and ledger statistics."""
    gateway = Gateway(_load(args.config))
    print(json.dumps(gateway.stats(), indent=2, ensure_ascii=False))
    return 0


def _cmd_providers(args: argparse.Namespace) -> int:
    """List configured providers, their types, and default models."""
    config = _load(args.config)
    print(f"{'PROVIDER':<16}{'TYPE':<16}{'DEFAULT MODEL':<28}{'TIMEOUT'}")
    print("-" * 70)
    for name in sorted(config.get("providers", {})):
        section = config["providers"][name]
        print(f"{name:<16}{str(section.get('type', '?')):<16}"
              f"{str(section.get('default_model') or '-'):<28}"
              f"{section.get('timeout_seconds', 60)}s")
    return 0


_COMMANDS = {
    "serve": _cmd_serve,
    "complete": _cmd_complete,
    "validate": _cmd_validate,
    "stats": _cmd_stats,
    "providers": _cmd_providers,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point; returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return _COMMANDS[args.command](args)


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
