"""Fetch large model artifacts from object storage at startup.

The big serialized artifacts (explainer/calibrator/stage2 .pkl, ~1.1G total)
exceed GitHub's 100MB per-file limit and bloat the Docker image, so they are
kept out of the repo and downloaded on container startup instead.

Configure URLs one of two ways (per-file env var wins over base URL):
  * Per file: set the env var named in each manifest entry's ``url_env``
    (e.g. EXPLAINER_URL, CALIBRATOR_URL, STAGE2_URL).
  * Base URL: set MODEL_BASE_URL; each file resolves to
    ``<MODEL_BASE_URL>/<filename>``.

Only HTTPS URLs are accepted (no code/secrets are uploaded; this only pulls
model weights down). Files already present with a matching SHA-256 are skipped,
so re-runs and local dev with pre-existing models are no-ops.
"""

import hashlib
import json
import os
import tempfile
import urllib.request
from typing import List, Optional


MANIFEST_NAME = "models_manifest.json"
_CHUNK = 1024 * 1024  # 1 MiB streaming chunks keep memory flat on big files


def _project_root() -> str:
    # src/api/model_loader.py -> project root is three levels up.
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def _resolve_url(entry: dict) -> Optional[str]:
    """Determine the download URL for a manifest entry, or None if unset."""
    env_name = entry.get("url_env")
    if env_name and os.environ.get(env_name):
        return os.environ[env_name].strip()

    base = os.environ.get("MODEL_BASE_URL", "").strip()
    if base:
        return f"{base.rstrip('/')}/{entry['filename']}"
    return None


def _download(url: str, dest: str) -> None:
    """Stream ``url`` to ``dest`` atomically (download to temp, then rename)."""
    if not url.lower().startswith("https://"):
        raise ValueError(f"Refusing non-HTTPS model URL: {url}")

    dest_dir = os.path.dirname(dest)
    os.makedirs(dest_dir, exist_ok=True)

    # Download to a temp file in the same dir so the final rename is atomic.
    fd, tmp_path = tempfile.mkstemp(dir=dest_dir, suffix=".part")
    os.close(fd)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "symptotriage-model-loader"})
        with urllib.request.urlopen(req) as resp, open(tmp_path, "wb") as out:
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp_path, dest)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def ensure_models(models_dir: str = "models") -> List[str]:
    """Ensure all manifest-listed artifacts exist locally with valid checksums.

    Returns the list of filenames that were downloaded (empty if all were
    already present and valid). Raises RuntimeError if a required file is
    missing and no URL is configured, or if a downloaded file fails its
    checksum.
    """
    root = _project_root()
    manifest_path = os.path.join(root, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        # No manifest => nothing to fetch (e.g. all artifacts shipped locally).
        return []

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if not os.path.isabs(models_dir):
        models_dir = os.path.join(root, models_dir)

    downloaded: List[str] = []
    for entry in manifest.get("files", []):
        filename = entry["filename"]
        expected = entry.get("sha256")
        dest = os.path.join(models_dir, filename)

        # Already present and valid? Skip.
        if os.path.exists(dest):
            if not expected or _sha256(dest) == expected:
                continue
            print(f"[model_loader] {filename} checksum mismatch; re-downloading.")

        url = _resolve_url(entry)
        if not url:
            raise RuntimeError(
                f"Model artifact '{filename}' is missing and no download URL is "
                f"configured. Set {entry.get('url_env')} or MODEL_BASE_URL."
            )

        print(f"[model_loader] Downloading {filename} from object storage...")
        _download(url, dest)

        if expected:
            actual = _sha256(dest)
            if actual != expected:
                os.remove(dest)
                raise RuntimeError(
                    f"Checksum mismatch for {filename}: expected {expected}, "
                    f"got {actual}. Deleted corrupt download."
                )
        downloaded.append(filename)
        print(f"[model_loader] {filename} ready.")

    return downloaded
