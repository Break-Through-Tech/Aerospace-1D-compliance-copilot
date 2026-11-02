"""JSONL cache helpers shared by pipeline runners."""
import json
from pathlib import Path
from .schemas import AuditRecord
from .prompts import SYSTEM_PROMPT, PROMPT_VERSION

def _cache_key(judge, embedder_name: str, k: int, req_text: str) -> str:
    import hashlib
    raw = "|".join([judge.name, str(getattr(judge, "model", "")), embedder_name, str(k),
                    SYSTEM_PROMPT, PROMPT_VERSION, req_text])
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def load_cache(path):
    cache = {}
    if path and Path(path).exists():
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                    record = AuditRecord.model_validate(entry["record"])
                    if record.verdict is not None and record.grounded:
                        cache[entry["key"]] = record
                except (ValueError, KeyError, TypeError):
                    continue  # tolerate an interrupted final write
        print(f"loaded {len(cache)} cached verdicts from {path}")
    return cache


def append_record(path, key, record):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"key": key, "record": record.model_dump(mode="json")}) + "\n")
