"""JSON configuration loading with environment-variable interpolation."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_ROUTING_STRATEGIES = ("priority", "round_robin", "cheapest", "latency")


class ConfigError(Exception):
    """Raised when the configuration file is missing, malformed, or invalid."""


def interpolate(value: Any, env: Dict[str, str]) -> Any:
    """Recursively replace ``${VAR}`` placeholders in strings with env values."""
    if isinstance(value, str):
        def _replace(match: "re.Match[str]") -> str:
            name = match.group(1)
            if name not in env:
                raise ConfigError(f"environment variable '{name}' is not set")
            return env[name]
        return _ENV_PATTERN.sub(_replace, value)
    if isinstance(value, dict):
        return {key: interpolate(item, env) for key, item in value.items()}
    if isinstance(value, list):
        return [interpolate(item, env) for item in value]
    return value


def load_config(path: str, env: Dict[str, str] = None) -> Dict[str, Any]:
    """Load and validate a gateway configuration file."""
    env = env if env is not None else dict(os.environ)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise ConfigError(f"config file not found: {path}")
    except json.JSONDecodeError as exc:
        raise ConfigError(f"config file is not valid JSON: {exc}")
    if not isinstance(raw, dict):
        raise ConfigError("config root must be a JSON object")
    config = interpolate(raw, env)
    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """Validate the structural invariants of a loaded configuration."""
    providers = config.get("providers")
    if not isinstance(providers, dict) or not providers:
        raise ConfigError("'providers' must be a non-empty object")
    for name, provider in providers.items():
        if not isinstance(provider, dict):
            raise ConfigError(f"providers.{name} must be an object")
        if not provider.get("type") or not isinstance(provider["type"], str):
            raise ConfigError(f"providers.{name}.type must be a non-empty string")

    routing = config.get("routing")
    if not isinstance(routing, dict):
        raise ConfigError("'routing' must be an object")
    strategy = routing.get("strategy")
    if strategy not in _ROUTING_STRATEGIES:
        raise ConfigError(
            "routing.strategy must be one of " + ", ".join(_ROUTING_STRATEGIES))
    candidates = routing.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ConfigError("'routing.candidates' must be a non-empty list")
    for index, candidate in enumerate(candidates):
        if candidate not in providers:
            raise ConfigError(
                f"routing.candidates[{index}] references unknown provider '{candidate}'")

    attempts = routing.get("max_attempts", 3)
    if not isinstance(attempts, int) or attempts < 1:
        raise ConfigError("routing.max_attempts must be an integer >= 1")
