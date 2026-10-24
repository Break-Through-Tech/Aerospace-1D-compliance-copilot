"""
Compliance Copilot - October milestone
Non-Agentic RAG Baseline & Schema Enforcement

Pipeline (deterministic, no agent loop):
    requirement text
      -> embed -> FAISS top-k clause retrieval (in-memory index over NPR 7150.2D clauses)
      -> LLM judge, forced into a Pydantic schema (tool-use structured output)
      -> grounding validation (cited clause_id MUST be one of the retrieved clauses)
      -> evaluation vs. the ground-truth benchmark (precision / recall / F1, citation groundedness)

Usage (Colab or local):
    pip install -r requirements-october.txt
    # Optional providers: pip install anthropic google-genai sentence-transformers
    export ANTHROPIC_API_KEY=...    # or GEMINI_API_KEY=... with --judge gemini
    # fully local: --embedder ollama --judge ollama  (needs `ollama serve` + models pulled)

    python scripts/compliance_rag_baseline.py \
        --clauses data/npr-7150-2d-requirements.csv \
        --benchmark data/npr7150_ground_truth_benchmark.csv \
        --embedder sbert --judge claude --k 5

    # offline smoke test (no API key / no model download):
    python scripts/compliance_rag_baseline.py ... --embedder tfidf --judge heuristic

Outputs (in --out, default ./out):
    predictions.csv   one row per benchmark requirement (gold vs predicted, cited clause, rationale)
    metrics.json      all metrics
    report.md         human-readable summary + confusion matrix + misses

Design notes:
  * The judge never sees the gold label, gold clause_id, or swe_id of the benchmark row.
    The model must pick the best-matching clause FROM THE RETRIEVED SET, so retrieval quality
    and judgement quality are both measured honestly.
  * Temperature 0 + forced tool call = reproducible, schema-valid output.
  * Any verdict that cites a clause outside the retrieved set is rejected and retried; if it still
    fails it is recorded as an invalid (ungrounded) prediction and counted in the groundedness metric.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


# =====================================================================
# 1. Schemas (Pydantic) - these force structured citations and verdicts
# =====================================================================
class Verdict(str, Enum):
    MEETS = "Meets"
    PARTIAL = "Partial"
    GAP = "Gap"


class Clause(BaseModel):
    clause_id: str
    swe_id: str
    text: str

    def as_context(self) -> str:
        return f"Clause ID: {self.clause_id}\nSWE ID: {self.swe_id}\nText: {self.text}"


class AuditVerdict(BaseModel):
    """The ONLY output shape the judge is allowed to produce.
    Field ORDER matters: structured decoding fills fields top to bottom, so the model must write the
    clause choice, evidence and missing elements BEFORE it commits to a verdict (reason first, decide last)."""
    clause_id: str = Field(description="Clause number ONLY of the single retrieved clause this verdict is about, "
                                       "e.g. '4.1.2'. No SWE ID, no other text.")
    evidence: str = Field(description="Short phrase copied from the CLAUSE text naming the obligation being checked.")
    missing_elements: List[str] = Field(
        default_factory=list,
        description="Each element the CLAUSE requires that the requirement does NOT address, or addresses only "
                    "vaguely. Never list things the requirement adds beyond the clause. Empty ONLY if every "
                    "element of the clause is covered.")
    rationale: str = Field(min_length=15, description="1-2 sentences comparing the requirement to the clause wording, "
                                                      "justifying the missing_elements and the verdict.")
    verdict: Verdict = Field(description="Decide LAST, from missing_elements: Meets = nothing missing; "
                                         "Partial = the clause's intent is addressed but something it requires is "
                                         "missing or vague; Gap = the clause's intent is not met.")

    @field_validator("clause_id")
    @classmethod
    def _norm_clause(cls, v: str) -> str:
        v = re.sub(r"^(clause( id)?|cl\.?)[:\s]*", "", v.strip(), flags=re.I)
        m = re.match(r"\d+(?:\.\d+)*", v)          # tolerate '4.1.2 | SWE-050' -> '4.1.2' (formatting only)
        return m.group(0) if m else v

    @model_validator(mode="after")
    def _meets_has_no_gaps(self):
        if self.verdict == Verdict.MEETS and self.missing_elements:
            raise ValueError("verdict is Meets but missing_elements is non-empty")
        if self.verdict != Verdict.MEETS and not self.missing_elements:
            raise ValueError("Partial/Gap verdicts must list at least one missing element")
        return self


class AuditRecord(BaseModel):
    requirement_id: str
    requirement_text: str
    retrieved_clause_ids: List[str]
    verdict: Optional[AuditVerdict] = None
    grounded: bool = False          # cited clause_id is in retrieved set
    error: Optional[str] = None


# =====================================================================
# 2. Data loading
# =====================================================================
def load_clauses(path: str) -> List[Clause]:
    df = pd.read_csv(path, dtype=str).fillna("")
    return [Clause(clause_id=r.clause.strip(), swe_id=r.swe_id.strip(), text=r.requirement.strip())
            for r in df.itertuples()]


def load_benchmark(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    df["clause_id"] = df["clause_id"].str.strip()
    df["label"] = df["label"].str.strip().str.capitalize()
    bad = set(df["label"]) - {v.value for v in Verdict}
    if bad:
        raise ValueError(f"Unexpected labels in benchmark: {bad}")
    return df


# =====================================================================
# 3. Embedding + in-memory vector index (FAISS)
# =====================================================================
class TfidfEmbedder:
    """Offline fallback (no model download). Fit on the clause corpus; dense L2-normalised vectors."""
    name = "tfidf"

    def __init__(self, corpus: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True).fit(corpus)

    def encode(self, texts: List[str]) -> np.ndarray:
        x = self.vec.transform(texts).toarray().astype("float32")
        n = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(n, 1e-9)


class SbertEmbedder:
    name = "sbert"

    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)

    def encode(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype("float32")


class OllamaEmbedder:
    """Local embeddings through Ollama (default nomic-embed-text). No API key."""
    name = "ollama"

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None):
        from ollama import Client
        self.client = Client(host=host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
        self.model = model or os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

    def encode(self, texts: List[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 64):                       # batch to keep requests small
            r = self.client.embed(model=self.model, input=texts[i:i + 64], keep_alive="30m")
            out.extend(r.embeddings)
        x = np.asarray(out, dtype="float32")
        return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


class ClauseIndex:
    """FAISS inner-product index (cosine, since vectors are normalised). Falls back to numpy if faiss missing."""

    def __init__(self, clauses: List[Clause], embedder):
        self.clauses, self.embedder = clauses, embedder
        mat = embedder.encode([c.text for c in clauses])
        try:
            import faiss
            self.index = faiss.IndexFlatIP(mat.shape[1])
            self.index.add(mat)
            self._np = None
        except ImportError:
            self.index, self._np = None, mat

    def search(self, query: str, k: int) -> List[tuple[Clause, float]]:
        q = self.embedder.encode([query])
        k = min(k, len(self.clauses))
        if self.index is not None:
            scores, idx = self.index.search(q, k)
            scores, idx = scores[0], idx[0]
        else:
            s = (self._np @ q[0])
            idx = np.argsort(-s)[:k]
            scores = s[idx]
        return [(self.clauses[i], float(s)) for i, s in zip(idx, scores)]


# =====================================================================
# 4. Judges
# =====================================================================
SYSTEM_PROMPT = """You are a compliance auditor checking a software requirement against NASA NPR 7150.2D clauses.

You are given ONE requirement and a set of RETRIEVED clauses. Work in this order:

STEP 1 - Choose the clause. Pick the single retrieved clause whose main obligation (its subject and its action)
is most directly what the requirement is about. Prefer the most specific match, not a general one.

STEP 2 - Break the clause into the elements it requires (who/what must be done, and any conditions such as
approval, tracing, maintenance, verification criteria).

STEP 3 - Check each element against what the requirement actually says.
  * An element counts as covered only if the requirement states it or clearly implies it.
  * Put every uncovered or vaguely covered element in missing_elements.
  * Anything the requirement adds BEYOND the clause (extra reviews, extra rigor, extra steps) is never a
    problem and never lowers the verdict. Do not list it as missing.
  * Judge only against the clause wording. Do not use outside knowledge of the standard.

STEP 4 - Decide the verdict from missing_elements:
  * Meets   : every element is covered (missing_elements is empty).
  * Partial : the requirement addresses the clause's intent but one or more required elements are missing or vague.
  * Gap     : the clause's intent is not met - it contradicts the clause, is unverifiable/informal, or does
              not address the obligation at all.

clause_id must be ONLY the clause number exactly as shown after 'Clause ID:' (e.g. 4.1.2) - never include the
SWE ID. You may ONLY cite a clause that appears in the retrieved set.
Submit your answer by calling the submit_verdict tool."""


PROMPT_VERSION = "v2-reasoning-first-rules"   # bump when the prompt/context format changes (invalidates cache)


def build_user_prompt(req_text: str, hits: List[tuple[Clause, float]]) -> str:
    ctx = "\n\n".join(c.as_context() for c, _ in hits)
    return f"RETRIEVED CLAUSES:\n{ctx}\n\nREQUIREMENT UNDER AUDIT:\n{req_text}"


class ClaudeJudge:
    """Forced tool-use => the API returns JSON matching AuditVerdict's schema. Validated with Pydantic."""
    name = "claude"

    def __init__(self, model: Optional[str] = None, max_retries: int = 2):
        import anthropic
        self.client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
        self.model = model or os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
        self.max_retries = max_retries
        self.tool = {
            "name": "submit_verdict",
            "description": "Submit the structured compliance verdict for this requirement.",
            "input_schema": AuditVerdict.model_json_schema(),
        }

    def judge(self, req_text: str, hits: List[tuple[Clause, float]]) -> AuditVerdict:
        allowed = {c.clause_id for c, _ in hits}
        messages = [{"role": "user", "content": build_user_prompt(req_text, hits)}]
        last_err = None
        for attempt in range(self.max_retries + 1):
            resp = self._call(messages)
            block = next((b for b in resp.content if b.type == "tool_use"), None)
            if block is None:
                last_err = "no tool call returned"
            else:
                try:
                    v = AuditVerdict.model_validate(block.input)
                    if v.clause_id not in allowed:
                        raise ValueError(f"cited clause {v.clause_id!r} not in retrieved set {sorted(allowed)}")
                    return v
                except (ValidationError, ValueError) as e:
                    last_err = str(e)
                    messages += [
                        {"role": "assistant", "content": resp.content},
                        {"role": "user", "content": [{
                            "type": "tool_result", "tool_use_id": block.id, "is_error": True,
                            "content": f"Invalid output: {last_err}. Fix it and call submit_verdict again."}]},
                    ]
        raise RuntimeError(f"judge failed after retries: {last_err}")

    def _call(self, messages):
        delay = 2.0
        for _ in range(5):                            # simple backoff for rate limits / overloads
            try:
                return self.client.messages.create(
                    model=self.model, max_tokens=800, temperature=0, system=SYSTEM_PROMPT,
                    tools=[self.tool], tool_choice={"type": "tool", "name": "submit_verdict"},
                    messages=messages)
            except Exception as e:                    # noqa: BLE001
                if "rate" in str(e).lower() or "overloaded" in str(e).lower() or "529" in str(e):
                    time.sleep(delay)
                    delay *= 2
                else:
                    raise
        raise RuntimeError("API retries exhausted")


class GeminiJudge:
    """Gemini with JSON-schema-constrained output (response_schema=AuditVerdict), validated with Pydantic.
    Same grounding check + retry-with-feedback loop as ClaudeJudge."""
    name = "gemini"

    def __init__(self, model: Optional[str] = None, max_retries: int = 2):
        from google import genai
        from google.genai import types
        self.types = types
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY)")
        self.client = genai.Client(api_key=key)
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.max_retries = max_retries

    def judge(self, req_text: str, hits: List[tuple[Clause, float]]) -> AuditVerdict:
        allowed = {c.clause_id for c, _ in hits}
        prompt = build_user_prompt(req_text, hits)
        last_err = None
        for _ in range(self.max_retries + 1):
            text = self._call(prompt if last_err is None else
                              f"{prompt}\n\nYour previous answer was invalid: {last_err}. Return corrected JSON.")
            try:
                v = AuditVerdict.model_validate_json(text)
                if v.clause_id not in allowed:
                    raise ValueError(f"cited clause {v.clause_id!r} not in retrieved set {sorted(allowed)}")
                return v
            except (ValidationError, ValueError) as e:
                last_err = str(e)[:400]
        raise RuntimeError(f"judge failed after retries: {last_err}")

    def _call(self, prompt: str) -> str:
        cfg = self.types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.replace("calling the submit_verdict tool", "returning JSON"),
            temperature=0, response_mime_type="application/json", response_schema=AuditVerdict)
        delay = 5.0
        for attempt in range(5):                      # backoff for per-minute rate limits (429)
            try:
                return self.client.models.generate_content(model=self.model, contents=prompt, config=cfg).text
            except Exception as e:                    # noqa: BLE001
                msg = str(e)
                if any(x in msg for x in ("PerDay", "per day", "daily")):
                    raise                              # daily quota: waiting will not help
                if any(x in msg for x in ("429", "RESOURCE_EXHAUSTED", "503", "overloaded")):
                    print(f"  rate limited, waiting {delay:.0f}s (attempt {attempt + 1}/5)...", flush=True)
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                else:
                    raise
        raise RuntimeError("API retries exhausted (rate limited). Increase --sleep or retry later.")


class OllamaJudge:
    """Local LLM through Ollama (default qwen3:4b). Output is constrained to AuditVerdict's JSON schema via
    Ollama's `format=` structured outputs, then validated with Pydantic + grounding check, with retry-on-failure."""
    name = "ollama"

    def __init__(self, model: Optional[str] = None, host: Optional[str] = None, max_retries: int = 2,
                 num_ctx: int = 4096):
        from ollama import Client
        self.client = Client(host=host or os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434"))
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen3:4b")
        self.max_retries, self.num_ctx = max_retries, num_ctx

    def judge(self, req_text: str, hits: List[tuple[Clause, float]]) -> AuditVerdict:
        allowed = {c.clause_id for c, _ in hits}
        system = SYSTEM_PROMPT.replace("calling the submit_verdict tool", "returning JSON that matches the schema")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": build_user_prompt(req_text, hits)}]
        last_err = None
        for _ in range(self.max_retries + 1):
            resp = self.client.chat(
                model=self.model, messages=messages, format=AuditVerdict.model_json_schema(),
                options={"temperature": 0, "num_ctx": self.num_ctx, "num_predict": 600},
                think=False, stream=False, keep_alive="30m")
            text = resp.message.content
            try:
                v = AuditVerdict.model_validate_json(text)
                if v.clause_id not in allowed:
                    raise ValueError(f"cited clause {v.clause_id!r} not in retrieved set {sorted(allowed)}")
                return v
            except (ValidationError, ValueError) as e:
                last_err = str(e)[:400]
                messages += [{"role": "assistant", "content": text},
                             {"role": "user", "content": f"Invalid output: {last_err}. Allowed clause_id values: "
                                                         f"{sorted(allowed)}. Return corrected JSON only."}]
        raise RuntimeError(f"judge failed after retries: {last_err}")


class HeuristicJudge:
    """Offline pipeline smoke-test ONLY (not a real baseline): top-1 clause, verdict from similarity + vague words.
    Lets you verify retrieval, schema, grounding and metrics code without an API key."""
    name = "heuristic"
    VAGUE = re.compile(r"\b(manual|manually|spreadsheet|no formal|without|ad hoc|informal|as needed|"
                       r"best effort|when possible|optional)\b", re.I)

    def judge(self, req_text, hits):
        clause, score = hits[0]
        if self.VAGUE.search(req_text) or score < 0.15:
            return AuditVerdict(clause_id=clause.clause_id, verdict=Verdict.GAP,
                                rationale="Requirement is informal or weakly related to the clause wording.",
                                evidence=clause.text[:60], missing_elements=["formal compliance with clause"])
        if score < 0.35:
            return AuditVerdict(clause_id=clause.clause_id, verdict=Verdict.PARTIAL,
                                rationale="Requirement overlaps only part of the clause wording.",
                                evidence=clause.text[:60], missing_elements=["remaining clause elements"])
        return AuditVerdict(clause_id=clause.clause_id, verdict=Verdict.MEETS,
                            rationale="Requirement closely matches the clause wording.",
                            evidence=clause.text[:60], missing_elements=[])


# =====================================================================
# 5. Pipeline
# =====================================================================
def _cache_key(judge, embedder_name: str, k: int, req_text: str) -> str:
    import hashlib
    raw = "|".join([judge.name, str(getattr(judge, "model", "")), embedder_name, str(k),
                    SYSTEM_PROMPT, PROMPT_VERSION, req_text])
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def run_pipeline(bench: pd.DataFrame, index: ClauseIndex, judge, k: int,
                 cache_path: Optional[str] = None, sleep: float = 0.0) -> List[AuditRecord]:
    """Resumable: every successful row is appended to cache_path (jsonl). Re-running skips cached rows,
    so a Colab disconnect / quota stop loses nothing. Cache key includes judge, model, embedder, k, prompt."""
    cache = {}
    if cache_path and os.path.exists(cache_path):
        for line in open(cache_path, encoding="utf-8"):
            try:
                d = json.loads(line)
                cache[d["key"]] = AuditRecord.model_validate(d["record"])
            except Exception:                                   # noqa: BLE001
                pass
        print(f"loaded {len(cache)} cached verdicts from {cache_path}")

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
        if rec.verdict is not None and cache_path:
            with open(cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "record": rec.model_dump(mode="json")}) + "\n")
        records.append(rec)
        status = rec.verdict.verdict.value if rec.verdict else f"ERROR: {rec.error}"
        print(f"[{i}/{len(bench)}] req {rec.requirement_id}: {status}", flush=True)
        if sleep:
            time.sleep(sleep)
    return records


# =====================================================================
# 6. Evaluation
# =====================================================================
def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=round(p, 4), recall=round(r, 4), f1=round(f, 4), tp=tp, fp=fp, fn=fn)


def evaluate(bench: pd.DataFrame, records: List[AuditRecord], k: int):
    df = bench.copy().reset_index(drop=True)
    df["pred_verdict"] = [r.verdict.verdict.value if r.verdict else "INVALID" for r in records]
    df["pred_clause"] = [r.verdict.clause_id if r.verdict else "" for r in records]
    df["pred_rationale"] = [r.verdict.rationale if r.verdict else (r.error or "") for r in records]
    df["pred_missing"] = [" | ".join(r.verdict.missing_elements) if r.verdict else "" for r in records]
    df["retrieved"] = [",".join(r.retrieved_clause_ids) for r in records]
    df["grounded"] = [r.grounded for r in records]
    df["hit_at_k"] = [g in r.retrieved_clause_ids for g, r in zip(df.clause_id, records)]
    df["hit_at_1"] = [r.retrieved_clause_ids[0] == g for g, r in zip(df.clause_id, records)]
    df["clause_match"] = df.clause_id == df.pred_clause
    df["correct"] = df.label == df.pred_verdict

    gold_gap, pred_gap = df.label == "Gap", df.pred_verdict == "Gap"
    gold_nc, pred_nc = df.label != "Meets", ~df.pred_verdict.isin(["Meets"]) & (df.pred_verdict != "INVALID")

    labels = ["Meets", "Partial", "Gap"]
    cm = pd.crosstab(df.label, df.pred_verdict).reindex(index=labels, columns=labels + ["INVALID"], fill_value=0)
    per_class = {c: prf(int(((df.label == c) & (df.pred_verdict == c)).sum()),
                        int(((df.label != c) & (df.pred_verdict == c)).sum()),
                        int(((df.label == c) & (df.pred_verdict != c)).sum())) for c in labels}

    n = len(df)
    metrics = dict(
        n=n, k=k,
        retrieval=dict(hit_at_1=round(df.hit_at_1.mean(), 4), **{f"hit_at_{k}": round(df.hit_at_k.mean(), 4)}),
        verdict_accuracy=round(df.correct.mean(), 4),
        gap_detection=prf(int((gold_gap & pred_gap).sum()), int((~gold_gap & pred_gap).sum()),
                          int((gold_gap & ~pred_gap).sum())),
        noncompliance_detection_gap_or_partial=prf(int((gold_nc & pred_nc).sum()), int((~gold_nc & pred_nc).sum()),
                                                   int((gold_nc & ~pred_nc).sum())),
        per_class=per_class,
        citation_groundedness=round(df.grounded.mean(), 4),
        cited_gold_clause_rate=round(df.clause_match.mean(), 4),
        schema_valid_rate=round((df.pred_verdict != "INVALID").mean(), 4),
        confusion_matrix=cm.to_dict(),
    )
    return df, metrics, cm


def write_report(path: Path, metrics: dict, cm: pd.DataFrame, df: pd.DataFrame, cfg: dict):
    g, nc, r = metrics["gap_detection"], metrics["noncompliance_detection_gap_or_partial"], metrics["retrieval"]
    kk = metrics["k"]
    lines = [
        "# October Milestone - Non-Agentic RAG Baseline Results", "",
        f"Embedder: `{cfg['embedder']}` | Judge: `{cfg['judge']}` | top-k: {kk} | n = {metrics['n']}", "",
        "## Headline metrics", "",
        "| Metric | Value |", "|---|---|",
        f"| Retrieval hit@1 | {r['hit_at_1']:.2%} |",
        f"| Retrieval hit@{kk} | {r[f'hit_at_{kk}']:.2%} |",
        f"| 3-way verdict accuracy | {metrics['verdict_accuracy']:.2%} |",
        f"| **Gap precision** | {g['precision']:.2%} |",
        f"| **Gap recall** | {g['recall']:.2%} |",
        f"| Gap F1 | {g['f1']:.2%} |",
        f"| Non-compliance (Gap+Partial) precision / recall | {nc['precision']:.2%} / {nc['recall']:.2%} |",
        f"| Citation groundedness | {metrics['citation_groundedness']:.2%} |",
        f"| Cited the gold clause | {metrics['cited_gold_clause_rate']:.2%} |",
        f"| Schema-valid outputs | {metrics['schema_valid_rate']:.2%} |", "",
        "## Confusion matrix (rows = human label, cols = predicted)", "", cm.to_markdown(), "",
        "## Errors (for the error-analysis section)", "",
        "| req | gold | pred | gold clause | cited | retrieved? |", "|---|---|---|---|---|---|",
    ]
    for x in df[~df.correct].itertuples():
        lines.append(f"| {x.requirement_id} | {x.label} | {x.pred_verdict} | {x.clause_id} | "
                     f"{x.pred_clause or '-'} | {'yes' if x.hit_at_k else 'NO'} |")
    path.write_text("\n".join(lines), encoding="utf-8")


# =====================================================================
# 7. CLI
# =====================================================================
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--clauses", required=True)
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--embedder", choices=["sbert", "ollama", "tfidf"], default="sbert")
    ap.add_argument("--sbert-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--ollama-embed-model", default=None, help="default: $OLLAMA_EMBED_MODEL or nomic-embed-text")
    ap.add_argument("--judge", choices=["claude", "gemini", "ollama", "heuristic"], default="claude")
    ap.add_argument("--model", default=None, help="judge model id (default: $CLAUDE_MODEL / $GEMINI_MODEL / $OLLAMA_MODEL=qwen3:4b)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None, help="only first N benchmark rows (cheap dry run)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--cache", default=None, help="jsonl cache for resumable runs (default: <out>/cache.jsonl)")
    ap.add_argument("--sleep", type=float, default=0.0,
                    help="seconds to pause between API calls (free-tier Gemini: try 6)")
    a = ap.parse_args(argv)

    clauses = load_clauses(a.clauses)
    bench = load_benchmark(a.benchmark)
    if a.limit:
        bench = bench.head(a.limit)
    missing = set(bench.clause_id) - {c.clause_id for c in clauses}
    if missing:
        print(f"WARNING: benchmark clause_ids not in clause corpus: {sorted(missing)}")

    embedder = {"sbert": lambda: SbertEmbedder(a.sbert_model),
                "ollama": lambda: OllamaEmbedder(a.ollama_embed_model),
                "tfidf": lambda: TfidfEmbedder([c.text for c in clauses])}[a.embedder]()
    index = ClauseIndex(clauses, embedder)
    judge = {"claude": lambda: ClaudeJudge(a.model), "gemini": lambda: GeminiJudge(a.model), "ollama": lambda: OllamaJudge(a.model),
             "heuristic": HeuristicJudge}[a.judge]()

    Path(a.out).mkdir(parents=True, exist_ok=True)
    records = run_pipeline(bench, index, judge, a.k, a.cache or str(Path(a.out) / "cache.jsonl"), a.sleep)
    df, metrics, cm = evaluate(bench, records, a.k)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "predictions.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    write_report(out / "report.md", metrics, cm, df, dict(embedder=a.embedder, judge=a.judge))
    (out / "audit_records.json").write_text(
        json.dumps([r.model_dump(mode="json") for r in records], indent=2))

    print("\n=== SUMMARY ===")
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("confusion_matrix", "per_class")}, indent=2))
    print(f"\nSaved to {out.resolve()}/ (predictions.csv, metrics.json, report.md, audit_records.json)")
    return df, metrics


if __name__ == "__main__":
    main()
