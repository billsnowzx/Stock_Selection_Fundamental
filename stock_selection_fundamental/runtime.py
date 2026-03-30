from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import uuid

from .config import ConfigBundle


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
