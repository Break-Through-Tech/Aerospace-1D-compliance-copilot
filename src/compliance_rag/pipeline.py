"""Retrieve, judge, validate grounding, and resume completed requirements."""
import time
from typing import List, Optional
import pandas as pd
from .schemas import AuditRecord
from .retrieval import ClauseIndex
from .cache import _cache_key, load_cache, append_record

def run_pipeline(bench: pd.DataFrame, index: ClauseIndex, judge, k: int,
                 cache_path: Optional[str] = None, sleep: float = 0.0) -> List[AuditRecord]:
    """Resumable: every successful row is appended to cache_path (jsonl). Re-running skips cached rows,
    so a Colab disconnect / quota stop loses nothing. Cache key includes judge, model, embedder, k, prompt."""
    cache = load_cache(cache_path)

    records = []
    for i, row in enumerate(bench.itertuples(), 1):
        key = _cache_key(judge, index.embedder.name, k, row.requirement_text)
        if key in cache:
            rec = cache[key].model_copy(update={"requirement_id": str(row.requirement_id)})
            records.append(rec)
            print(f"[{i}/{len(bench)}] req {rec.requirement_id}: {rec.verdict.verdict.value} (cached)", flush=True)
            continue
        hits = index.search(row.requirement_text, k)           # NOTE: only requirement text is used
        rec = AuditRecord(requirement_id=str(row.requirement_id), requirement_text=row.requirement_text,
                          retrieved_clause_ids=[c.clause_id for c, _ in hits])
        try:
            rec.verdict = judge.judge(row.requirement_text, hits)
            rec.grounded = rec.verdict.clause_id in rec.retrieved_clause_ids
        except Exception as e:                                  # noqa: BLE001
            rec.error = str(e)[:300]
            # config / quota problems would fail every remaining row: stop (progress is already cached)
            if any(x in str(e) for x in ("404", "NOT_FOUND", "401", "403", "PERMISSION_DENIED",
                                         "UNAUTHENTICATED", "API key", "authentication",
                                         "PerDay", "per day", "daily", "DAILY",
                                         "Connection refused", "ConnectError", "not found")):
                raise SystemExit(f"\nFATAL on row {i} - stopping (finished rows are saved; rerun to resume):\n{e}\n"
                                 "Check the model name, API key / quota, or that the Ollama server is running and the model is pulled.")
        if rec.verdict is not None and rec.grounded and cache_path:
            append_record(cache_path, key, rec)
        records.append(rec)
        status = rec.verdict.verdict.value if rec.verdict else f"ERROR: {rec.error}"
        print(f"[{i}/{len(bench)}] req {rec.requirement_id}: {status}", flush=True)
        if sleep:
            time.sleep(sleep)
    return records
