from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import hashlib
import json
import uuid

from .config import ConfigBundle


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def generate_run_id(prefix: str = "run") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return f"{prefix}_{ts}_{short}"


def hash_data_dir(data_dir: str | Path) -> str:
    path = Path(data_dir)
    digest = hashlib.sha256()
    if not path.exists():
        return ""
    for file in sorted(path.glob("*.csv")):
        digest.update(file.name.encode("utf-8"))
        stat = file.stat()
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(int(stat.st_mtime)).encode("utf-8"))
    return digest.hexdigest()


def hash_config_bundle(bundle: ConfigBundle) -> str:
    digest = hashlib.sha256()
    payload = json.dumps(bundle.as_dict(), sort_keys=True, ensure_ascii=False, default=str)
    digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


def append_jsonl(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str))
        handle.write("\n")
    return target


def append_run_audit(
    output_root: str | Path,
    payload: dict[str, Any],
    filename: str = "run_audit.jsonl",
) -> Path:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    enriched = dict(payload)
    enriched.setdefault("logged_at", utc_now_iso())
    return append_jsonl(root / filename, enriched)
