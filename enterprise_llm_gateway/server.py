"""A dependency-free HTTP server exposing an OpenAI-compatible surface.

Routes:
- ``POST /v1/chat/completions`` the OpenAI-compatible completion endpoint

``GET  /healthz``            liveness probe
``GET  /v1/models``          lists routable providers and their default models
``GET  /metrics``            Prometheus text exposition of gateway internals
``POST /v1/chat/completions`` the OpenAI-compatible completion endpoint
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple

from .gateway import Gateway
from .models import ModelRequest
from .providers.base import ProviderError
from .router import RoutingError

_MAX_BODY_BYTES = 4 * 1024 * 1024
_SAFE_MODEL_NAME = re.compile(r"[^A-Za-z0-9_.\-:]+")


class GatewayHandler(BaseHTTPRequestHandler):
    """Request handler bound to a :class:`Gateway` via ``server.gateway``."""

    server_version = "EnterpriseLLMGateway/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing ---------------------------------------------------------

    @property
    def gateway(self) -> Gateway:
        """The gateway instance attached to the running server."""
        return self.server.gateway  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        """Route access logs to stderr only when the server is verbose."""
        if getattr(self.server, "verbose", False):  # type: ignore[attr-defined]
            super().log_message(fmt, *args)

    def _send_json(self, status: int, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, status: int, body: str,
                   content_type: str = "text/plain; charset=utf-8") -> None:
        payload = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _read_json(self) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        length_header = self.headers.get("Content-Length")
        try:
            length = int(length_header) if length_header else 0
        except ValueError:
            return None, "invalid Content-Length header"
        if length <= 0:
            return None, "request body is required"
        if length > _MAX_BODY_BYTES:
            return None, f"request body exceeds {_MAX_BODY_BYTES} bytes"
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            return None, f"invalid JSON body: {exc}"
        if not isinstance(data, dict):
            return None, "request body must be a JSON object"
        return data, None

    # -- routes -----------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        """Serve health, model listing, and metrics endpoints."""
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path == "/healthz":
            self._send_json(200, {
                "status": "ok",
                "uptime_seconds": round(time.time() - self.server.started_at, 3),  # type: ignore[attr-defined]
            })
        elif path == "/v1/models":
            self._send_json(200, self._list_models())
        elif path == "/metrics":
            self._send_text(200, self._render_metrics(),
                            "text/plain; version=0.0.4; charset=utf-8")
        else:
            self._send_json(404, {"error": {"message": f"no route for {path}",
                                            "type": "not_found"}})

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        """Serve the OpenAI-compatible chat completions endpoint."""
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path != "/v1/chat/completions":
            self._send_json(404, {"error": {"message": f"no route for {path}",
                                            "type": "not_found"}})
            return

        data, error = self._read_json()
        if error is not None:
            self._send_json(400, {"error": {"message": error, "type": "invalid_request_error"}})
            return

        try:
            request = self._build_request(data)
        except ValueError as exc:
            self._send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            return

        try:
            result = self.gateway.complete(request)
        except RoutingError as exc:
            self._send_json(503, {"error": {"message": str(exc), "type": "routing_error",
                                            "attempts": exc.attempts}})
            return
        except ProviderError as exc:
            self._send_json(502, {"error": {"message": exc.message, "type": "provider_error",
                                            "provider": exc.provider}})
            return

        payload = result.response.to_openai_json()
        payload["provider"] = result.provider
        payload["x_gateway"] = {
            "cache_hit": result.cached,
            "attempts": result.attempts,
            "latency_ms": round(result.total_latency_ms, 3),
        }
        self._send_json(200, payload)

    def _build_request(self, data: Dict[str, Any]) -> ModelRequest:
        """Translate an OpenAI request body into a :class:`ModelRequest`."""
        messages = data.get("messages")
        prompt = data.get("prompt", "")
        if not messages and not prompt:
            raise ValueError("either 'messages' or 'prompt' is required")
        if isinstance(messages, list):
            for message in messages:
                if not isinstance(message, dict):
                    raise ValueError("each entry in 'messages' must be an object")
        stop = data.get("stop")
        if isinstance(stop, str):
            stop = [stop]
        if stop is not None and not isinstance(stop, list):
            raise ValueError("'stop' must be a string or a list of strings")
        return ModelRequest(
            prompt=str(prompt or ""),
            messages=messages,
            model=data.get("model"),
            temperature=data.get("temperature"),
            max_tokens=data.get("max_tokens"),
            stop=stop,
            top_p=data.get("top_p"),
            metadata=data.get("metadata") or {},
            user=data.get("user"),
        )

    def _list_models(self) -> Dict[str, Any]:
        """Build the ``/v1/models`` payload from provider defaults."""
        models = []
        for name, provider in sorted(self.gateway.providers.items()):
            models.append({
                "id": f"{name}:{provider.default_model or 'default'}",
                "object": "model",
                "owned_by": provider.__class__.__name__,
                "provider": name,
            })
        return {"object": "list", "data": models}

    def _render_metrics(self) -> str:
        """Render gateway counters in Prometheus text exposition format."""
        stats = self.gateway.stats()
        cache = stats["cache"]
        ledger = stats["ledger"]
        lines: list = []

        def emit(name: str, value: float, labels: str = "",
                 kind: str = "gauge") -> None:
            """Append one metric to the exposition buffer.

            Args:
                name: Metric name.
                value: Numeric sample value.
                labels: Pre-formatted ``{key="value"}`` label text; may be empty.
                kind: Metric type, for example ``gauge`` or ``counter``.

            Returns:
                A single exposition line including the trailing newline.
            """
            lines.append(f"# TYPE {name} {kind}")
            lines.append(f"{name}{labels} {value}")

        emit("gateway_requests_total", ledger.get("requests", 0), kind="counter")
        emit("gateway_cache_hits_total", ledger.get("cache_hits", 0), kind="counter")
        emit("gateway_errors_total", ledger.get("errors", 0), kind="counter")
        emit("gateway_cache_hit_rate", cache.get("hit_rate", 0.0))
        emit("gateway_cache_entries", cache.get("entries", 0))
        emit("gateway_total_cost_usd", ledger.get("total_cost_usd", 0.0))
        emit("gateway_prompt_tokens_total", ledger.get("prompt_tokens", 0), kind="counter")
        emit("gateway_completion_tokens_total", ledger.get("completion_tokens", 0), kind="counter")
        emit("gateway_p95_latency_ms", ledger.get("p95_latency_ms", 0.0))

        for provider in stats.get("provider_health", []):
            label = 'provider="{}"'.format(_SAFE_MODEL_NAME.sub("_", provider["provider"]))
            emit("gateway_provider_requests_total", provider["calls"], label, "counter")
            emit("gateway_provider_failures_total", provider["failures"], label, "counter")
            emit("gateway_provider_latency_ewma_ms", provider["latency_ewma_ms"], label)
        return "\n".join(lines) + "\n"


class GatewayServer:

    """Wraps a :class:`Gateway` in an ``http.server`` handler.

Args:
        gateway: The gateway to serve.
        host: Bind address; defaults to loopback so the server is not exposed
            to the network unless an operator opts in.
        port: Bind port; ``0`` picks an ephemeral port.
        verbose: Emit per-request access logs to stderr.
    """

    def __init__(self, gateway: Gateway, host: str = "127.0.0.1", port: int = 8080,
                 verbose: bool = False) -> None:
        self.gateway = gateway
        self._httpd = ThreadingHTTPServer((host, port), GatewayHandler)
        self._httpd.gateway = gateway  # type: ignore[attr-defined]
        self._httpd.verbose = verbose  # type: ignore[attr-defined]
        self._httpd.started_at = time.time()  # type: ignore[attr-defined]
        self._thread: Optional[threading.Thread] = None

    @property
    def address(self) -> Tuple[str, int]:
        """Return the bound ``(host, port)`` after startup."""
        return self._httpd.server_address[0], self._httpd.server_address[1]

    def serve_forever(self) -> None:
        """Block and serve requests until interrupted."""
        self._httpd.serve_forever()

    def start_background(self) -> Tuple[str, int]:
        """Serve on a daemon thread and return the bound address."""
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self.address

    def shutdown(self) -> None:
        """Stop the server and release the listening socket."""
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
