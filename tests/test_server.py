"""HTTP-level tests against a live ephemeral gateway server.

The server binds to port 0 on loopback, so these tests exercise the real
request path without binding a fixed port or contacting any provider.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

from enterprise_llm_gateway.server import GatewayServer


@pytest.fixture
def live_server(make_gateway):
    """Start a gateway server on an ephemeral port and yield its base URL."""
    gateway = make_gateway()
    server = GatewayServer(gateway, host="127.0.0.1", port=0)
    host, port = server.start_background()
    try:
        yield f"http://{host}:{port}", gateway
    finally:
        server.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _post(url, payload):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, json.loads(response.read().decode("utf-8"))


def _post_raw(url, body: bytes):
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=5) as response:
        return response.status, response.read().decode("utf-8")


def test_healthz(live_server):
    base, _ = live_server
    status, body = _get(f"{base}/healthz")
    assert status == 200
    assert body["status"] == "ok"
    assert body["uptime_seconds"] >= 0


def test_models_lists_providers(live_server):
    base, _ = live_server
    status, body = _get(f"{base}/v1/models")
    assert status == 200
    assert body["object"] == "list"
    ids = {entry["id"] for entry in body["data"]}
    assert "primary:fake-model" in ids


def test_metrics_is_prometheus_text(live_server):
    base, _ = live_server
    with urllib.request.urlopen(f"{base}/metrics", timeout=5) as response:
        assert response.status == 200
        assert response.headers["Content-Type"].startswith("text/plain")
        text = response.read().decode("utf-8")
    assert "gateway_cache_hit_rate" in text
    assert "gateway_total_cost_usd" in text


def test_completion_returns_openai_shape(live_server):
    base, gateway = live_server
    gateway.cache.enabled = False
    status, body = _post(f"{base}/v1/chat/completions",
                         {"messages": [{"role": "user", "content": "hello"}]})
    assert status == 200
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "ok"
    assert body["provider"] == "primary"
    assert body["x_gateway"]["cache_hit"] is False


def test_completion_accepts_prompt_field(live_server):
    base, gateway = live_server
    gateway.cache.enabled = False
    status, body = _post(f"{base}/v1/chat/completions", {"prompt": "hi"})
    assert status == 200
    assert body["choices"][0]["message"]["content"] == "ok"


def test_second_identical_call_hits_cache(live_server):
    base, gateway = live_server
    gateway.cache.enabled = True
    payload = {"messages": [{"role": "user", "content": "same"}]}
    _post(f"{base}/v1/chat/completions", payload)
    _, body = _post(f"{base}/v1/chat/completions", payload)
    assert body["x_gateway"]["cache_hit"] is True


def test_string_stop_is_normalised(live_server):
    base, gateway = live_server
    gateway.cache.enabled = False
    status, body = _post(f"{base}/v1/chat/completions",
                         {"prompt": "hi", "stop": "END"})
    assert status == 200
    assert body["choices"][0]["message"]["content"] == "ok"


def test_missing_body_field_is_rejected(live_server):
    base, _ = live_server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(f"{base}/v1/chat/completions", {"model": "gpt"})
    assert excinfo.value.code == 400
    payload = json.loads(excinfo.value.read().decode("utf-8"))
    assert "messages" in payload["error"]["message"]


def test_invalid_json_is_rejected(live_server):
    base, _ = live_server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post_raw(f"{base}/v1/chat/completions", b"{not json")
    assert excinfo.value.code == 400
    payload = json.loads(excinfo.value.read().decode("utf-8"))
    assert "invalid JSON" in payload["error"]["message"]


def test_non_object_json_is_rejected(live_server):
    base, _ = live_server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post_raw(f"{base}/v1/chat/completions", b"[1,2,3]")
    assert excinfo.value.code == 400


def test_bad_stop_type_is_rejected(live_server):
    base, _ = live_server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(f"{base}/v1/chat/completions", {"prompt": "hi", "stop": 42})
    assert excinfo.value.code == 400


def test_unknown_route_returns_404(live_server):
    base, _ = live_server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _get(f"{base}/nope")
    assert excinfo.value.code == 404


def test_routing_failure_returns_503(make_gateway, base_config):
    import copy

    config = copy.deepcopy(base_config)
    for name in ("primary", "secondary"):
        config["providers"][name]["type"] = "fake_boom"
        config["providers"][name]["status_code"] = 503
    gateway = make_gateway(config)
    server = GatewayServer(gateway, host="127.0.0.1", port=0)
    host, port = server.start_background()
    try:
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            _post(f"http://{host}:{port}/v1/chat/completions", {"prompt": "hi"})
        assert excinfo.value.code == 503
        payload = json.loads(excinfo.value.read().decode("utf-8"))
        assert payload["error"]["type"] == "routing_error"
    finally:
        server.shutdown()


def test_post_to_unknown_route_returns_404(live_server):
    base, _ = live_server
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        _post(f"{base}/v1/embeddings", {"input": "hi"})
    assert excinfo.value.code == 404
