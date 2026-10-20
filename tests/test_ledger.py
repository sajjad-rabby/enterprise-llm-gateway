"""Tests for the append-only usage and cost ledger."""

from __future__ import annotations

import json
import os

import pytest

from enterprise_llm_gateway.ledger import LedgerRecord, UsageLedger


def _record(provider="openai", cost=0.01, prompt=100, completion=50,
            cache_hit=False, error=None, user=None, latency=120.0):
    return LedgerRecord(
        timestamp=1700000000.0,
        request_id="req-1",
        provider=provider,
        model="m",
        user=user,
        prompt_tokens=prompt,
        completion_tokens=completion,
        cost_usd=cost,
        latency_ms=latency,
        cache_hit=cache_hit,
        error=error,
    )


def test_record_to_dict_shape():
    data = _record().to_dict()
    assert data["provider"] == "openai"
    assert data["cost_usd"] == 0.01
    assert data["cache_hit"] is False


def test_record_roundtrips_through_dict():
    original = _record(user="alice", error=None)
    restored = LedgerRecord.from_dict(original.to_dict())
    assert restored.provider == original.provider
    assert restored.user == "alice"
    assert restored.prompt_tokens == original.prompt_tokens


def test_usage_totals_accumulate():
    ledger = UsageLedger()
    ledger.record(_record(cost=0.02))
    ledger.record(_record(cost=0.03))
    summary = ledger.summary()
    assert summary["requests"] == 2
    assert summary["total_cost_usd"] == 0.05
    assert summary["prompt_tokens"] == 200
    assert summary["completion_tokens"] == 100


def test_cache_hits_counted_but_not_billed():
    ledger = UsageLedger()
    ledger.record(_record(cost=0.05))
    ledger.record(_record(cost=0.0, cache_hit=True))
    summary = ledger.summary()
    assert summary["cache_hits"] == 1
    assert summary["total_cost_usd"] == 0.05


def test_saved_amount_reflects_cache_avoided_cost():
    ledger = UsageLedger()
    ledger.record(_record(cost=0.05))
    # A cache hit carries what the call would have cost, not what it did.
    ledger.record(_record(cost=0.05, cache_hit=True))
    ledger.record(_record(cost=0.05, cache_hit=True))
    summary = ledger.summary()
    assert summary["gross_cost_usd"] == 0.15
    assert summary["total_cost_usd"] == 0.05
    assert summary["saved_usd"] == pytest.approx(0.10)
    assert summary["cache_hits"] == 2


def test_cache_hits_do_not_appear_in_provider_buckets():
    ledger = UsageLedger()
    ledger.record(_record(provider="openai", cost=0.05))
    ledger.record(_record(provider="openai", cost=0.05, cache_hit=True))
    bucket = ledger.summary()["by_provider"]["openai"]
    assert bucket["requests"] == 1
    assert bucket["cost_usd"] == 0.05


def test_summary_groups_by_provider():
    ledger = UsageLedger()
    ledger.record(_record(provider="openai", cost=0.01))
    ledger.record(_record(provider="anthropic", cost=0.02))
    ledger.record(_record(provider="anthropic", cost=0.03))
    by_provider = ledger.summary()["by_provider"]
    assert by_provider["openai"]["requests"] == 1
    assert by_provider["anthropic"]["requests"] == 2
    assert by_provider["anthropic"]["cost_usd"] == 0.05


def test_errors_are_counted_per_provider():
    ledger = UsageLedger()
    ledger.record(_record(provider="openai", cost=0.0, error="routing_exhausted"))
    summary = ledger.summary()
    assert summary["errors"] == 1
    assert summary["by_provider"]["openai"]["errors"] == 1


def test_percentiles_are_computed():
    ledger = UsageLedger()
    for latency in range(1, 101):
        ledger.record(_record(latency=float(latency)))
    summary = ledger.summary()
    assert summary["p50_latency_ms"] == 50.5
    assert 95 <= summary["p95_latency_ms"] <= 96


def test_percentiles_are_zero_when_empty():
    summary = UsageLedger().summary()
    assert summary["p50_latency_ms"] == 0.0
    assert summary["p95_latency_ms"] == 0.0


def test_records_persist_to_jsonl(tmp_path):
    path = str(tmp_path / "nested" / "ledger.jsonl")
    ledger = UsageLedger(path=path)
    ledger.record(_record(cost=0.04))
    ledger.record(_record(cost=0.06, provider="anthropic"))
    assert os.path.exists(path)
    lines = open(path, encoding="utf-8").read().strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["cost_usd"] == 0.04


def test_load_returns_persisted_records(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger = UsageLedger(path=path)
    ledger.record(_record(cost=0.01))
    ledger.record(_record(cost=0.02, provider="azure"))
    loaded = ledger.load()
    assert len(loaded) == 2
    assert sum(r.cost_usd for r in loaded) == 0.03


def test_load_missing_file_returns_empty(tmp_path):
    assert UsageLedger(path=str(tmp_path / "absent.jsonl")).load() == []


def test_load_skips_malformed_lines(tmp_path):
    path = tmp_path / "ledger.jsonl"
    path.write_text('{"provider": "openai"}\nnot json\n\n', encoding="utf-8")
    records = UsageLedger(path=str(path)).load()
    assert len(records) == 1
    assert records[0].provider == "openai"


def test_memory_retention_is_bounded():
    ledger = UsageLedger(retain_in_memory=5)
    for _ in range(20):
        ledger.record(_record())
    assert len(ledger._recent) == 5
    assert ledger._total_requests == 20


def test_summary_falls_back_to_recent_records_when_not_persisted():
    ledger = UsageLedger(retain_in_memory=100)
    for _ in range(3):
        ledger.record(_record(cost=0.01))
    assert ledger.summary()["requests"] == 3
    assert ledger.summary()["total_cost_usd"] == pytest.approx(0.03)
