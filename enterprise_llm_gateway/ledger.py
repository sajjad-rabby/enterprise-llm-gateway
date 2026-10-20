"""Append-only usage ledger backed by JSONL.

Records are never mutated, so every summary reflects

Every request that reaches a provider is recorded as one JSON line so
spend can be attributed per provider, model, and caller without needing a
database. The ledger is intentionally append-only: records are immutable
facts about what was billed, and summaries are computed on read.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Dict, Iterable, List, Optional


class LedgerRecord:
    """A single billed request."""

    __slots__ = ("timestamp", "request_id", "provider", "model", "user",
                 "prompt_tokens", "completion_tokens", "cost_usd",
                 "latency_ms", "cache_hit", "error")

    def __init__(self, timestamp: float, request_id: str, provider: str,
                 model: str, user: Optional[str], prompt_tokens: int,
                 completion_tokens: int, cost_usd: float, latency_ms: float,
                 cache_hit: bool = False, error: Optional[str] = None) -> None:
        self.timestamp = timestamp
        self.request_id = request_id
        self.provider = provider
        self.model = model
        self.user = user
        self.prompt_tokens = int(prompt_tokens)
        self.completion_tokens = int(completion_tokens)
        self.cost_usd = float(cost_usd)
        self.latency_ms = float(latency_ms)
        self.cache_hit = bool(cache_hit)
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Render the record as a JSON-serializable mapping."""
        return {
            "timestamp": self.timestamp,
            "request_id": self.request_id,
            "provider": self.provider,
            "model": self.model,
            "user": self.user,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_ms": round(self.latency_ms, 3),
            "cache_hit": self.cache_hit,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LedgerRecord":
        """Rebuild a record from its persisted mapping."""
        return cls(
            timestamp=float(data.get("timestamp", 0.0)),
            request_id=str(data.get("request_id", "")),
            provider=str(data.get("provider", "")),
            model=str(data.get("model", "")),
            user=data.get("user"),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            cost_usd=float(data.get("cost_usd", 0.0)),
            latency_ms=float(data.get("latency_ms", 0.0)),
            cache_hit=bool(data.get("cache_hit", False)),
            error=data.get("error"),
        )


class UsageLedger:

    """Collects ledger records in memory and on disk.

Args:
        path: File to append records to. ``None`` keeps records in memory only.
        retain_in_memory: How many recent records to keep for fast summaries.
    """

    def __init__(self, path: Optional[str] = None, retain_in_memory: int = 1000) -> None:
        self.path = path
        self.retain_in_memory = int(retain_in_memory)
        self._recent: List[LedgerRecord] = []
        self._lock = threading.Lock()
        self._total_cost = 0.0
        self._total_requests = 0
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            if directory:
                os.makedirs(directory, exist_ok=True)

    def record(self, record: LedgerRecord) -> LedgerRecord:
        """Persist a record and fold it into the running totals."""
        with self._lock:
            self._total_cost += record.cost_usd
            self._total_requests += 1
            if not record.cache_hit and record.error is None:
                pass
            self._recent.append(record)
            if len(self._recent) > self.retain_in_memory:
                del self._recent[0]
        if self.path:
            self._append(record)
        return record

    def _append(self, record: LedgerRecord) -> None:
        line = json.dumps(record.to_dict(), ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def load(self) -> List[LedgerRecord]:
        """Read every persisted record, skipping malformed lines."""
        if not self.path or not os.path.exists(self.path):
            return []
        records: List[LedgerRecord] = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(LedgerRecord.from_dict(json.loads(line)))
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
        return records

    def summary(self, records: Optional[Iterable[LedgerRecord]] = None) -> Dict[str, Any]:
        """Summarise records by token totals, cost and cache effectiveness.

Args:
    records: Records to summarise; defaults to the retained in-memory window.

The hypothetical cost of a cache hit is reported alongside the

        Cache hits are counted but contribute no cost, which makes the
        saved amount visible directly in the summary.
        """
        source = list(records) if records is not None else self.load() or self._recent
        by_provider: Dict[str, Dict[str, Any]] = {}
        total_cost = 0.0
        gross_cost = 0.0
        prompt_tokens = 0
        completion_tokens = 0
        cache_hits = 0
        errors = 0
        latencies: List[float] = []

        for record in source:
            prompt_tokens += record.prompt_tokens
            completion_tokens += record.completion_tokens
            latencies.append(record.latency_ms)
            provider_key = record.provider or "unknown"
            if record.error:
                errors += 1
                # Failed requests still belong to a provider's error
                # tally, even though they contribute no cost or tokens.
                by_provider.setdefault(
                    provider_key,
                    {"requests": 0, "cost_usd": 0.0, "prompt_tokens": 0,
                     "completion_tokens": 0, "errors": 0},
                )["errors"] += 1
                continue

            # Every non-error record contributes to the gross figure,
            # including cache hits, which carry the price the cached
            # response *would* have cost. Net spend excludes cache hits.
            gross_cost += record.cost_usd
            if record.cache_hit:
                cache_hits += 1
                continue
            total_cost += record.cost_usd
            bucket = by_provider.setdefault(
                provider_key,
                {"requests": 0, "cost_usd": 0.0, "prompt_tokens": 0,
                 "completion_tokens": 0, "errors": 0},
            )
            bucket["requests"] += 1
            bucket["cost_usd"] += record.cost_usd
            bucket["prompt_tokens"] += record.prompt_tokens
            bucket["completion_tokens"] += record.completion_tokens

        for bucket in by_provider.values():
            bucket["cost_usd"] = round(bucket["cost_usd"], 6)

        latencies.sort()
        return {
            "requests": len(source),
            "cache_hits": cache_hits,
            "errors": errors,
            "total_cost_usd": round(total_cost, 6),
            "gross_cost_usd": round(gross_cost, 6),
            "saved_usd": round(gross_cost - total_cost, 6),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "p50_latency_ms": _percentile(latencies, 0.50),
            "p95_latency_ms": _percentile(latencies, 0.95),
            "by_provider": by_provider,
        }


def _percentile(sorted_values: List[float], fraction: float) -> float:
    """Return a linear-interpolated percentile of a pre-sorted list."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return round(sorted_values[0], 3)
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return round(sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight, 3)
