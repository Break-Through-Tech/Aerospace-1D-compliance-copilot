"""Small, single-writer JSONL cache for successful canonical audits."""

import hashlib
import json
import os
from pathlib import Path

from src.schemas import AuditRecord


def cache_key(value) -> str:
    """Hash JSON deterministically, independently of process or dictionary order."""
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def successful(record: AuditRecord) -> bool:
    return (
        record.verdict is not None
        and record.grounded
        and record.error is None
        and record.verdict.clause_id in record.retrieved_clause_ids
    )


def load_cache(path: str | Path) -> dict[str, AuditRecord]:
    """Validate entries and repair only a torn final write before future appends.

    Complete malformed lines and schema errors propagate. A valid final record
    without a newline is kept, with its separator restored.
    """
    try:
        handle = Path(path).open("r+b")
    except FileNotFoundError:
        return {}
    records = {}
    with handle:
        while True:
            start = handle.tell()
            line = handle.readline()
            if not line:
                break
            try:
                entry = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b"\n"):
                    raise
                handle.truncate(start)
                break
            if not isinstance(entry, dict) or set(entry) != {"key", "record"}:
                raise ValueError("Cache entry must contain key and record")
            if not isinstance(entry["key"], str) or not entry["key"]:
                raise ValueError("Cache key must be a nonempty string")
            record = AuditRecord.model_validate(entry["record"], strict=True)
            if successful(record):
                records[entry["key"]] = record
            if not line.endswith(b"\n"):
                handle.write(b"\n")
    return records


def append_record(path: str | Path, key: str, record: AuditRecord) -> None:
    """Persist each success before the next requirement can fail."""
    if not successful(record):
        return
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    entry = {"key": key, "record": record.model_dump(mode="json")}
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def client_settings(client) -> dict:
    """Capture actual injected model settings, excluding transport and tracing."""
    return client.model_dump(mode="json", exclude={
        "callbacks", "tags", "metadata", "verbose", "name", "cache",
        "client_kwargs", "sync_client_kwargs", "async_client_kwargs",
    })


def index_fingerprint(vector_store) -> str:
    """Read stored contents once; sorting makes Chroma's return order irrelevant.

    Include vectors because re-embedding can change retrieval without changing
    text. LangChain exposes contents via get(), but collection configuration
    (including distance metric) requires its underlying Chroma collection.
    """
    contents = vector_store.get(include=["documents", "metadatas", "embeddings"])
    rows = [
        {
            "id": item_id,
            "document": contents["documents"][i],
            "metadata": contents["metadatas"][i],
            "embedding": list(map(float, contents["embeddings"][i])),
        }
        for i, item_id in enumerate(contents["ids"])
    ]
    collection = vector_store._collection
    return cache_key({
        "rows": sorted(rows, key=lambda row: row["id"]),
        "metadata": collection.metadata,
        "configuration": collection.configuration_json,
    })
