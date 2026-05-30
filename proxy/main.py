"""Appwrite Function: thin reverse proxy to the Render-hosted SymptoTriage API.

The ML backend runs as a warm process on Render (loads models once, SHAP works).
This function does NO ML — it just forwards each incoming HTTP request to the
Render service and relays the response. With no heavy imports or model loading,
it cold-starts well within the serverless readiness window.

Configure the backend with the RENDER_BACKEND_URL env var, e.g.
    RENDER_BACKEND_URL=https://symptotriage.onrender.com
"""

import os
import json
import urllib.request
import urllib.error

BACKEND = os.environ.get("RENDER_BACKEND_URL", "").rstrip("/")

# Hop-by-hop headers must not be forwarded.
_SKIP_REQ_HEADERS = {"host", "content-length", "connection", "x-appwrite-key"}
_SKIP_RESP_HEADERS = {"connection", "transfer-encoding", "content-encoding", "content-length"}


def main(context):
    if not BACKEND:
        return context.res.json(
            {"error": "RENDER_BACKEND_URL is not configured on this function."}, 500
        )

    req = context.req
    path = req.path or "/"
    qs = req.query_string or ""
    url = f"{BACKEND}{path}" + (f"?{qs}" if qs else "")

    method = (req.method or "GET").upper()
    body = req.body_binary if getattr(req, "body_binary", None) else None
    if isinstance(body, str):
        body = body.encode("utf-8")

    fwd_headers = {
        k: v for k, v in (req.headers or {}).items()
        if k.lower() not in _SKIP_REQ_HEADERS
    }

    request = urllib.request.Request(url, data=body, method=method, headers=fwd_headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as resp:
            payload = resp.read()
            status = resp.status
            headers = {
                k.lower(): v for k, v in resp.headers.items()
                if k.lower() not in _SKIP_RESP_HEADERS
            }
    except urllib.error.HTTPError as e:
        payload = e.read()
        status = e.code
        headers = {"content-type": e.headers.get("content-type", "application/json")}
    except Exception as e:
        context.error(f"Proxy error: {e}")
        return context.res.json({"error": "Upstream request failed", "detail": str(e)}, 502)

    return context.res.binary(payload, status, headers)
