"""Tests for configuration loading, interpolation, and validation."""

from __future__ import annotations

import json

import pytest

from enterprise_llm_gateway.config import (
    ConfigError,
    interpolate,
    load_config,
    validate_config,
)


def _write(tmp_path, data, name="config.json"):
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path)


def test_interpolate_replaces_known_variables():
    env = {"TOKEN": "secret"}
    assert interpolate("${TOKEN}", env) == "secret"
    assert interpolate("prefix-${TOKEN}-suffix", env) == "prefix-secret-suffix"


def test_interpolate_walks_nested_structures():
    env = {"HOST": "example.com"}
    payload = {"url": "https://${HOST}/v1", "list": ["${HOST}"], "n": 5}
    result = interpolate(payload, env)
    assert result["url"] == "https://example.com/v1"
    assert result["list"] == ["example.com"]
    assert result["n"] == 5


def test_interpolate_raises_on_missing_variable():
    with pytest.raises(ConfigError, match="ABSENT"):
        interpolate("${ABSENT}", {})


def test_load_config_missing_file(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(str(tmp_path / "nope.json"), {})


def test_load_config_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid JSON"):
        load_config(str(path), {})


def test_load_config_non_object_root(tmp_path):
    with pytest.raises(ConfigError, match="JSON object"):
        load_config(_write(tmp_path, [1, 2, 3]), {})


def test_load_config_success_and_env_substitution(tmp_path):
    path = _write(tmp_path, {
        "routing": {"strategy": "priority", "candidates": ["a"]},
        "providers": {"a": {"type": "openai", "api_key_env": "MY_KEY",
                            "api_key": "${MY_SECRET}"}},
    })
    config = load_config(path, {"MY_SECRET": "abc123"})
    assert config["providers"]["a"]["api_key"] == "abc123"


def test_validate_requires_providers():
    with pytest.raises(ConfigError, match="'providers'"):
        validate_config({"providers": {}, "routing": {"strategy": "priority",
                                                      "candidates": ["a"]}})


def test_validate_requires_provider_type():
    with pytest.raises(ConfigError, match="type"):
        validate_config({"providers": {"a": {}},
                         "routing": {"strategy": "priority", "candidates": ["a"]}})


def test_validate_rejects_unknown_strategy():
    with pytest.raises(ConfigError, match="routing.strategy"):
        validate_config({"providers": {"a": {"type": "openai"}},
                         "routing": {"strategy": "random", "candidates": ["a"]}})


def test_validate_rejects_unknown_candidate():
    with pytest.raises(ConfigError, match="unknown provider 'ghost'"):
        validate_config({"providers": {"a": {"type": "openai"}},
                         "routing": {"strategy": "priority", "candidates": ["ghost"]}})


def test_validate_rejects_bad_max_attempts():
    with pytest.raises(ConfigError, match="max_attempts"):
        validate_config({"providers": {"a": {"type": "openai"}},
                         "routing": {"strategy": "priority", "candidates": ["a"],
                                     "max_attempts": 0}})


def test_validate_accepts_valid_config():
    validate_config({"providers": {"a": {"type": "openai"},
                                   "b": {"type": "ollama"}},
                     "routing": {"strategy": "latency",
                                 "candidates": ["a", "b"], "max_attempts": 4}})
