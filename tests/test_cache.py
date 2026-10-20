"""Tests for the TTL response cache."""

from __future__ import annotations

import time

from enterprise_llm_gateway.cache import CacheEntry, ResponseCache
from enterprise_llm_gateway.models import ModelRequest, ModelResponse, Usage


def _response(text="hello"):
    return ModelResponse(content=text, model="m", provider="p", usage=Usage(1, 1))


def test_set_then_get_returns_same_response():
    cache = ResponseCache(ttl_seconds=60)
    response = _response()
    cache.set("k", response)
    assert cache.get("k") is response


def test_get_missing_key_returns_none():
    cache = ResponseCache(ttl_seconds=60)
    assert cache.get("absent") is None
    assert cache.stats()["misses"] == 1


def test_expired_entry_is_not_returned():
    cache = ResponseCache(ttl_seconds=0.05)
    cache.set("k", _response())
    time.sleep(0.08)
    assert cache.get("k") is None


def test_disabled_cache_never_stores():
    cache = ResponseCache(ttl_seconds=60, enabled=False)
    cache.set("k", _response())
    assert cache.get("k") is None
    assert len(cache) == 0


def test_non_positive_ttl_disables_caching():
    cache = ResponseCache(ttl_seconds=0)
    cache.set("k", _response())
    assert cache.get("k") is None


def test_eviction_respects_max_entries():
    cache = ResponseCache(ttl_seconds=60, max_entries=2)
    for index in range(3):
        cache.set(f"k{index}", _response(f"r{index}"))
        time.sleep(0.01)
    assert len(cache) == 2
    assert cache.stats()["evictions"] == 1
    assert cache.get("k0") is None


def test_overwrite_replaces_without_growing():
    cache = ResponseCache(ttl_seconds=60, max_entries=3)
    cache.set("k", _response("first"))
    cache.set("k", _response("second"))
    assert len(cache) == 1
    assert cache.get("k").content == "second"


def test_hit_rate_is_computed():
    cache = ResponseCache(ttl_seconds=60)
    cache.set("k", _response())
    cache.get("k")
    cache.get("missing")
    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5


def test_hit_rate_is_zero_with_no_lookups():
    assert ResponseCache().stats()["hit_rate"] == 0.0


def test_purge_expired_removes_only_expired():
    cache = ResponseCache(ttl_seconds=0.05)
    cache.set("old", _response())
    time.sleep(0.08)
    cache.set("new", _response())
    assert cache.purge_expired() == 1
    assert cache.get("new") is not None


def test_clear_empties_entries_but_keeps_counters():
    cache = ResponseCache(ttl_seconds=60)
    cache.set("k", _response())
    cache.get("k")
    cache.clear()
    assert len(cache) == 0
    assert cache.stats()["hits"] == 1


def test_lookup_and_store_use_request_key():
    cache = ResponseCache(ttl_seconds=60)
    request = ModelRequest(prompt="what is 2+2?")
    cache.store(request, _response("4"))
    assert cache.lookup(request).content == "4"


def test_different_metadata_shares_one_cache_entry():
    cache = ResponseCache(ttl_seconds=60)
    first = ModelRequest(prompt="hello", user="alice")
    second = ModelRequest(prompt="hello", user="bob")
    cache.store(first, _response("hi"))
    assert cache.lookup(second).content == "hi"
    assert len(cache) == 1


def test_cache_entry_expiry_flag():
    entry = CacheEntry(_response(), ttl_seconds=0.01)
    assert not entry.is_expired()
    time.sleep(0.02)
    assert entry.is_expired()


def test_concurrent_access_is_safe():
    import threading

    cache = ResponseCache(ttl_seconds=60, max_entries=200)
    errors = []

    def worker(index: int) -> None:
        try:
            for step in range(50):
                cache.set(f"k{index}-{step}", _response())
                cache.get(f"k{index}-{step}")
        except Exception as exc:  # pragma: no cover - only on real failure
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(cache) <= 200
