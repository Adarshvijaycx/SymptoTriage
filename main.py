"""Appwrite Function entrypoint.

Appwrite Functions don't run a long-lived ``uvicorn`` server; instead the
Open Runtimes Python image imports this module ONCE per container and calls
``main(context)`` for every request. We exploit that by:

  1. Loading the ML pipeline a single time at module import (cold start), so
     warm invocations are fast.
  2. Bridging the per-request ``context.req`` / ``context.res`` objects to the
     existing FastAPI ASGI app, so all routes (/predict, /symptoms, the served
     frontend, ...) work unchanged.

Writable storage in a function is ``/tmp``, so models live in ``/tmp/models``:
the 6 small artifacts shipped in the deployment bundle are copied there, and
the 3 large artifacts are downloaded from object storage by model_loader.
"""

import os
import sys
import shutil
import traceback

# Models are baked into the deployment bundle (next to this file) to avoid a
# multi-minute runtime download that exceeds Appwrite's cold-start window.
# Loading from this local dir is read-only, which predict.py/ModelService allow.
MODELS_DIR = os.environ.get("MODELS_DIR", os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "models"))

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# The Open Runtimes loader imports this entrypoint as a module, and the project
# root is not guaranteed to be on sys.path — so `import src.api...` fails with
# ModuleNotFoundError. Add the project root explicitly so the `src` package
# (which sits next to this file) is importable regardless of the runtime's cwd.
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

_init_error = None
_init_done = False
app = None


def _bootstrap():
    """One-time initialization: stage models and load the pipeline.

    Run lazily on the FIRST request (NOT at import time): Open Runtimes kills a
    worker whose module import blocks too long, and downloading ~1.1GB + loading
    the SHAP explainer takes minutes. Keeping import instant lets the worker
    start; the heavy init then runs inside the (async, 900s) first invocation.
    """
    global app

    os.environ["MODELS_DIR"] = MODELS_DIR
    # Serve the static frontend from the same function so the UI + API share an
    # origin (no CORS needed).
    os.environ.setdefault("SERVE_FRONTEND", "1")

    # Models are bundled in MODELS_DIR (baked into the deployment), so there is
    # no download/staging step. Load the pipeline directly.
    import src.api.main as api
    from src.api.predict import ModelService

    api.service = ModelService(models_dir=MODELS_DIR)
    app = api.app


def _ensure_init():
    """Run _bootstrap once; cache success/failure for subsequent calls."""
    global _init_done, _init_error
    if _init_done:
        return
    try:
        _bootstrap()
    except Exception as e:
        _init_error = e
        traceback.print_exc()
    finally:
        _init_done = True


async def _drive_asgi(scope, body: bytes):
    """Run one HTTP request through the FastAPI ASGI app, collect the response."""
    received = {"done": False}

    async def receive():
        if not received["done"]:
            received["done"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    result = {"status": 500, "headers": [], "body": b""}

    async def send(message):
        if message["type"] == "http.response.start":
            result["status"] = message["status"]
            result["headers"] = message.get("headers", [])
        elif message["type"] == "http.response.body":
            result["body"] += message.get("body", b"")

    await app(scope, receive, send)
    return result


async def main(context):
    # Lazy one-time init (models download + load) on first request.
    _ensure_init()
    if _init_error is not None:
        context.error(f"Model init failed: {_init_error}")
        return context.res.json(
            {"error": "Model initialization failed", "detail": str(_init_error)},
            503,
        )

    req = context.req
    headers = [
        (str(k).lower().encode("latin-1"), str(v).encode("latin-1"))
        for k, v in (req.headers or {}).items()
    ]
    path = req.path or "/"
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": (req.method or "GET").upper(),
        "scheme": req.scheme or "https",
        "path": path,
        "raw_path": path.encode("latin-1"),
        "query_string": (req.query_string or "").encode("latin-1"),
        "headers": headers,
        "server": (req.host or "localhost", req.port or 443),
        "client": ("127.0.0.1", 0),
        "root_path": "",
    }

    body = req.body_binary if req.body_binary is not None else b""

    try:
        resp = await _drive_asgi(scope, body)
    except Exception as e:
        context.error("".join(traceback.format_exception(e)))
        return context.res.json({"error": "Inference failed", "detail": str(e)}, 500)

    out_headers = {}
    for k, v in resp["headers"]:
        key = k.decode("latin-1").lower()
        # content-length is recomputed by the runtime; dropping it avoids mismatch.
        if key == "content-length":
            continue
        out_headers[key] = v.decode("latin-1")

    return context.res.binary(resp["body"], resp["status"], out_headers)
