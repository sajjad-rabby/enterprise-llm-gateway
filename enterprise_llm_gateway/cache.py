"""LRU response cache with TTL expiry.

Entries are evicted in least-recently-used order

The cache lets repeated identical prompts skip a provider call entirely,
which is the single largest cost lever in most LLM workloads. Entries
expire after a configurable TTL and are bounded by a maximum entry count
so memory stays predictable under sustained load.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict, Optional, Tuple

from .models import ModelResponse, cache_key_for_request


class CacheEntry:
    """A cached response plus its insertion and expiry timestamps."""

    __slots__ = ("response", "stored_at", "expires_at", "hits")

    def __init__(self, response: ModelResponse, ttl_seconds: float) -> None:
        self.response = response
        self.stored_at = time.time()
        self.expires_at = self.stored_at + ttl_seconds
        self.hits = 0

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Return True once the entry is past its TTL."""
        return (now if now is not None else time.time()) >= self.expires_at


class ResponseCache:

    """Thread-safe LRU cache mapping a request fingerprint to a cached response.

Args:
        ttl_seconds: Lifetime of a cached entry. Values <= 0 disable caching.
        max_entries: Maximum retained entries; the oldest is evicted first.
        enabled: Master switch, useful for turning the cache off in tests.
    """

    def __init__(self, ttl_seconds: float = 3600.0, max_entries: int = 1000,
                 enabled: bool = True) -> None:
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self.enabled = bool(enabled)
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str) -> Optional[ModelResponse]:
        """Return a live cached response for ``key``, or None."""
        if not self.enabled or self.ttl_seconds <= 0:
            return None
        now = time.time()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired(now):
                del self._entries[key]
                self._misses += 1
                return None
            entry.hits += 1
            self._hits += 1
            return entry.response

    def set(self, key: str, response: ModelResponse) -> None:
        """Store ``response`` under ``key``, evicting the oldest if full."""
        if not self.enabled or self.ttl_seconds <= 0:
            return
        with self._lock:
            if key in self._entries:
                del self._entries[key]
            self._entries[key] = CacheEntry(response, self.ttl_seconds)
            while len(self._entries) > self.max_entries:
                oldest = min(self._entries.items(), key=lambda item: item[1].stored_at)
                del self._entries[oldest[0]]
                self._evictions += 1

    def lookup(self, request: Any) -> Optional[ModelResponse]:
        """Convenience wrapper computing the cache key from a request."""
        return self.get(cache_key_for_request(request))

    def store(self, request: Any, response: ModelResponse) -> None:
        """Convenience wrapper computing the cache key from a request."""
        self.set(cache_key_for_request(request), response)

    def purge_expired(self) -> int:
        """Drop every expired entry and return how many were removed."""
        now = time.time()
        with self._lock:
            expired = [key for key, entry in self._entries.items() if entry.is_expired(now)]
            for key in expired:
                del self._entries[key]
            return len(expired)

    def clear(self) -> None:
        """Remove every entry without resetting counters."""
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> Dict[str, Any]:
        """Return a snapshot of cache counters and hit rate."""
        with self._lock:
            lookups = self._hits + self._misses
            return {
                "enabled": self.enabled,
                "ttl_seconds": self.ttl_seconds,
                "max_entries": self.max_entries,
                "entries": len(self._entries),
                "hits": self._hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "hit_rate": round(self._hits / lookups, 4) if lookups else 0.0,
            }
