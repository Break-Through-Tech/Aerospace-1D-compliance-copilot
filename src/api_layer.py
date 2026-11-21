"""
api_layer.py - request rate limiting, batching and API error handling.

Sections in this file
    1. Request rate limiter      (Venya)
    2. API feature handling      (Venya)   retries, backoff, Retry-After, error classification
    3. Batching logic            (Venya)
    4. Request cache             (Aaliyan) bounded, thread-safe memoization with optional TTL

Standalone: standard library only. Works with any provider (Claude, Gemini, Ollama, ...) because it only
needs a callable, e.g.  api.call(judge.judge, requirement_text, hits).

Quick use
    api = ApiClient(rpm=10)                                   # <= 10 real requests per minute, with retries
    verdict = api.call(judge.judge, req_text, hits)           # one rate-limited, retried request

    results = api.run_concurrent(rows, audit_one, workers=1, batch_size=10,
                                 on_batch_done=lambda n, res: print("batch", n, "done"))
    embeddings = api.batch_call(texts, embedder.encode, batch_size=64)   # one request per 64 texts
"""
from __future__ import annotations

import random
import copy
import hashlib
import inspect
import json
import math
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from functools import wraps
from typing import Any, Callable, Iterator, List, Optional, Sequence


# =====================================================================
# 1. REQUEST RATE LIMITER  (Venya)
# =====================================================================
class RateLimiter:
    """Spaces requests at least 60/rpm seconds apart. Thread-safe. rpm <= 0 (or None) disables it.

    Call wait() immediately before EVERY request, including retries, so the limit also holds when a
    request is re-sent.
    """

    def __init__(self, rpm: Optional[float] = 0, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.interval = 60.0 / rpm if rpm and rpm > 0 else 0.0
        self._clock, self._sleep = clock, sleep
        self._lock, self._next_slot = threading.Lock(), 0.0

    def wait(self) -> float:
        """Block until this request may be sent. Returns the seconds slept."""
        if not self.interval:
            return 0.0
        with self._lock:                       # reserve the next free time slot...
            now = self._clock()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.interval
        delay = slot - now
        if delay > 0:
            self._sleep(delay)                 # ...and sleep outside the lock so other threads can reserve theirs
            return delay
        return 0.0


# =====================================================================
# 2. API FEATURE HANDLING  (Venya)
# =====================================================================
class FatalAPIError(RuntimeError):
    """Bad key / permissions / unknown model / server down. Every request would fail, so stop the run.
    When raised from a batch method, .partial_results holds everything finished before the stop."""

    def __init__(self, *args):
        super().__init__(*args)
        self.partial_results: list = []


class DailyQuotaError(FatalAPIError):
    """Daily/monthly quota used up. Waiting seconds will not help: stop, then resume later (cache helps here)."""


QUOTA_MARKERS = ("PerDay", "per day", "daily", "DAILY")
FATAL_MARKERS = ("NOT_FOUND", "PERMISSION_DENIED", "UNAUTHENTICATED", "invalid x-api-key", "API key",
                 "authentication", "Connection refused", "ConnectError", "not found")
RETRY_MARKERS = ("rate limit", "rate_limit", "RESOURCE_EXHAUSTED", "overloaded", "timed out", "timeout",
                 "temporarily unavailable")
FATAL_STATUS = (401, 403, 404)
RETRY_STATUS = (408, 429, 500, 502, 503, 504, 529)


def _status(exc: Exception) -> Optional[int]:
    """HTTP status from the exception, however the SDK exposes it (anthropic, google-genai, httpx, ollama)."""
    for obj, attr in ((exc, "status_code"), (exc, "code"), (getattr(exc, "response", None), "status_code")):
        value = getattr(obj, attr, None) if obj is not None else None
        if isinstance(value, int):
            return value
    return None


def classify(exc: Exception) -> str:
    """'quota' | 'fatal' | 'retry' | 'other'. Order matters: a daily-quota 429 must NOT be retried."""
    msg = str(exc)
    if any(m in msg for m in QUOTA_MARKERS):
        return "quota"
    status = _status(exc)
    if status in FATAL_STATUS or any(m in msg for m in FATAL_MARKERS):
        return "fatal"
    if status in RETRY_STATUS or isinstance(exc, TimeoutError) or any(m in msg for m in RETRY_MARKERS):
        return "retry"
    return "other"                             # e.g. a validation error: not an API problem, re-raise unchanged


def retry_after_seconds(exc: Exception) -> Optional[float]:
    """Seconds the server told us to wait (Retry-After header), if the SDK exposes the response."""
    headers = getattr(getattr(exc, "response", None), "headers", None)
    if not headers:
        return None
    for key in ("retry-after", "Retry-After"):
        try:
            return float(headers[key])
        except (KeyError, TypeError, ValueError):
            continue
    return None


@dataclass
class ApiStats:
    requests: int = 0            # real requests sent (every attempt counts)
    retries: int = 0
    failures: int = 0            # calls that gave up
    rate_limit_wait_s: float = 0.0
    backoff_wait_s: float = 0.0


class ApiClient:
    """Wraps any callable with: rate limiting -> request -> classified retry with exponential backoff."""

    def __init__(self, rpm: Optional[float] = 0, retries: int = 5, base_delay: float = 2.0, cap: float = 60.0,
                 jitter: float = 0.2, limiter: Optional[RateLimiter] = None,
                 sleep: Callable[[float], None] = time.sleep, verbose: bool = True):
        self.limiter = limiter or RateLimiter(rpm)
        self.retries, self.base_delay, self.cap, self.jitter = max(1, retries), base_delay, cap, jitter
        self._sleep, self.verbose = sleep, verbose
        self.stats, self._stats_lock = ApiStats(), threading.Lock()

    def _bump(self, **deltas) -> None:
        with self._stats_lock:
            for name, amount in deltas.items():
                setattr(self.stats, name, getattr(self.stats, name) + amount)

    def call(self, fn: Callable, *args, label: str = "api", **kwargs):
        """One logical request. Retries transient errors (429/5xx/overloaded/timeouts), honouring Retry-After.
        Quota and fatal config errors raise immediately; anything else re-raises unchanged."""
        delay = self.base_delay
        for attempt in range(1, self.retries + 1):
            self._bump(rate_limit_wait_s=self.limiter.wait())          # throttle EVERY attempt
            self._bump(requests=1)
            try:
                return fn(*args, **kwargs)
            except Exception as exc:                                    # noqa: BLE001
                kind = classify(exc)
                if kind == "quota":
                    self._bump(failures=1)
                    raise DailyQuotaError(str(exc)[:300]) from exc
                if kind == "fatal":
                    self._bump(failures=1)
                    raise FatalAPIError(str(exc)[:300]) from exc
                if kind != "retry" or attempt == self.retries:
                    self._bump(failures=1)
                    raise
                wait = retry_after_seconds(exc) or delay
                wait = min(wait, self.cap) * (1 + random.uniform(-self.jitter, self.jitter))
                if self.verbose:
                    print(f"  [{label}] transient error, waiting {wait:.1f}s (attempt {attempt}/{self.retries})",
                          flush=True)
                self._bump(retries=1, backoff_wait_s=wait)
                self._sleep(wait)
                delay = min(delay * 2, self.cap)                        # 2s, 4s, 8s ... capped

    def summary(self) -> dict:
        s = self.stats
        return dict(requests=s.requests, retries=s.retries, failures=s.failures,
                    rate_limit_wait_s=round(s.rate_limit_wait_s, 1), backoff_wait_s=round(s.backoff_wait_s, 1))

    # =================================================================
    # 3. BATCHING LOGIC  (Venya)
    # =================================================================
    def batch_call(self, items: Sequence, batch_fn: Callable[[list], Sequence], batch_size: int = 32,
                   label: str = "batch") -> List["ItemResult"]:
        """True API batching: ONE request per chunk of items, e.g. embeddings -> batch_fn(list_of_texts).
        batch_fn must return one result per input, in order. A chunk that fails (non-fatal) marks only its own
        items as errors; results always line up with the input order."""
        results: List[ItemResult] = []
        for start, chunk in chunked(items, batch_size):
            try:
                out = self.call(batch_fn, chunk, label=label)
                if len(out) != len(chunk):
                    raise ValueError(f"batch_fn returned {len(out)} results for {len(chunk)} inputs")
                results += [ItemResult(start + i, item, value=v) for i, (item, v) in enumerate(zip(chunk, out))]
            except FatalAPIError as exc:
                exc.partial_results = results
                raise
            except Exception as exc:                                    # noqa: BLE001
                results += [ItemResult(start + i, item, error=str(exc)[:300]) for i, item in enumerate(chunk)]
        return results

    def run_concurrent(self, items: Sequence, item_fn: Callable[[Any], Any], workers: int = 1,
                       batch_size: Optional[int] = None, on_batch_done: Optional[Callable] = None,
                       label: str = "item") -> List["ItemResult"]:
        """One request per item (e.g. one LLM judgement each), processed in batches of batch_size with up to
        `workers` running at once. All workers share ONE rate limiter, so rpm holds no matter how many threads.
        - one failing item never stops the others (its ItemResult carries the error)
        - a fatal/quota error stops the run after the in-flight batch; .partial_results has what finished
        - on_batch_done(batch_number, results_of_that_batch) fires after each batch: progress / checkpointing
        - results are returned in input order"""
        items = list(items)
        batch_size = batch_size or len(items) or 1
        results: List[ItemResult] = []
        stop = threading.Event()
        fatal: List[FatalAPIError] = []

        def one(indexed):
            index, item = indexed
            if stop.is_set():
                return ItemResult(index, item, error="skipped: run stopped after a fatal API error")
            try:
                return ItemResult(index, item, value=self.call(item_fn, item, label=label))
            except FatalAPIError as exc:
                fatal.append(exc)
                stop.set()
                return ItemResult(index, item, error=str(exc)[:300])
            except Exception as exc:                                    # noqa: BLE001
                return ItemResult(index, item, error=str(exc)[:300])

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for number, (start, chunk) in enumerate(chunked(items, batch_size), 1):
                batch = list(pool.map(one, list(enumerate(chunk, start))))
                results += batch
                if on_batch_done:
                    on_batch_done(number, batch)
                if fatal:
                    fatal[0].partial_results = [r for r in results if r.ok]
                    raise fatal[0]
        return results


def chunked(items: Sequence, size: int) -> Iterator[tuple]:
    """Yield (start_index, chunk) pairs of at most `size` items."""
    if size < 1:
        raise ValueError("batch size must be >= 1")
    for start in range(0, len(items), size):
        yield start, list(items[start:start + size])


@dataclass
class ItemResult:
    index: int                       # position in the input list
    item: Any
    value: Any = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# =====================================================================
# 4. REQUEST CACHE  (Aaliyan)
# =====================================================================
# Where it plugs in: put the cache OUTSIDE api.call so a cache hit never reaches the rate limiter:
#
#       cached_judge = cache.wrap(lambda req, hits: api.call(judge.judge, req, hits))
#
# (wrapping the other way round would make cache hits still wait for a rate-limit slot.)

def _request_value(value):
    """Encode supported inputs without lossy str()/repr() fallbacks or type collisions."""
    kind = type(value)
    type_name = f"{kind.__module__}.{kind.__qualname__}"
    if isinstance(value, Enum):
        return ["enum", type_name, _request_value(value.value)]
    if value is None or kind in (bool, int, str):
        return [kind.__name__, value]
    if kind is float and math.isfinite(value):
        return ["float", value]
    if kind in (list, tuple):
        return [kind.__name__, [_request_value(item) for item in value]]
    if kind is dict:
        pairs = [[_request_value(key), _request_value(item)] for key, item in value.items()]
        return ["dict", sorted(pairs, key=lambda pair: json.dumps(pair[0], sort_keys=True))]
    if is_dataclass(value) and not isinstance(value, type):
        return ["dataclass", type_name,
                _request_value({field.name: getattr(value, field.name) for field in fields(value)})]
    # Supports the repository's Pydantic v2 Clause models without importing Pydantic.
    if callable(getattr(value, "model_dump", None)):
        return ["model", type_name, _request_value(value.model_dump(mode="python"))]
    raise TypeError(f"Unsupported cache input: {type_name}; convert it to plain Python values")


class RequestCache:
    """Process-local LRU cache for completed calls, with optional expiration.

    Wrap the rate-limited call once and reuse that wrapper. Concurrent identical
    requests share one in-flight call; different keys can run in parallel. Only
    returned values are cached, never exceptions. Wrap validation as well as the
    API call if invalid provider responses must not be saved.

    Inputs support plain Python values, dataclasses and Pydantic v2 models.
    Results must support deepcopy; each caller gets its own copy. A wrapper has
    a private scope by default. Explicit namespaces allow deliberate sharing;
    include provider/model/prompt/settings versions in context when sharing.
    No values or keys are persisted to disk; src.cache handles resumable audits.
    """

    def __init__(self, maxsize: int = 1024, ttl: Optional[float] = None,
                 clock: Callable[[], float] = time.monotonic):
        if type(maxsize) is not int or maxsize < 1:
            raise ValueError("maxsize must be a positive integer")
        if ttl is not None and (isinstance(ttl, bool) or not math.isfinite(ttl) or ttl <= 0):
            raise ValueError("ttl must be a positive finite number or None")
        self.maxsize, self.ttl, self._clock = maxsize, ttl, clock
        self._entries = OrderedDict()
        self._pending = {}
        self._lock = threading.Lock()
        self._generation = 0
        self._hits = self._misses = self._waits = self._evictions = self._expirations = 0

    def wrap(self, fn: Callable, *, namespace: Optional[str] = None, context: Any = None) -> Callable:
        """Memoize fn; keyword order, positional spelling and defaults do not affect keys.

        Context is captured now. Recreate the wrapper (or clear the cache) after
        changing a closed-over model, prompt, endpoint or generation setting.
        Explicit namespaces must describe equivalent operations and signatures.
        """
        if namespace is not None and (not isinstance(namespace, str) or not namespace.strip()):
            raise ValueError("namespace must be a nonempty string or None")
        scope = namespace if namespace is not None else object()
        settings = _request_value(context)
        signature = inspect.signature(fn)

        @wraps(fn)
        def cached(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            payload = json.dumps([settings, _request_value(dict(bound.arguments))],
                                 ensure_ascii=False, sort_keys=True, allow_nan=False)
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
            with self._lock:
                generation = self._generation
                key = (generation, scope, digest)
                if key in self._entries:
                    expires, value = self._entries[key]
                    if expires is None or self._clock() < expires:
                        self._entries.move_to_end(key)
                        self._hits += 1
                        return copy.deepcopy(value)
                    del self._entries[key]
                    self._expirations += 1
                future = self._pending.get(key)
                owner = future is None
                if owner:
                    future = self._pending[key] = Future()
                    self._misses += 1
                else:
                    self._waits += 1
            if not owner:
                return copy.deepcopy(future.result())
            try:
                value = copy.deepcopy(fn(*args, **kwargs))
                result = copy.deepcopy(value)
                with self._lock:
                    if generation == self._generation:
                        expires = None if self.ttl is None else self._clock() + self.ttl
                        self._entries[key] = (expires, value)
                        if len(self._entries) > self.maxsize:
                            self._entries.popitem(last=False)
                            self._evictions += 1
                    del self._pending[key]
            except BaseException as exc:
                with self._lock:
                    self._pending.pop(key, None)
                future.set_exception(exc)
                raise
            future.set_result(value)
            return result

        return cached

    def clear(self) -> None:
        """Invalidate entries; calls already running finish without repopulating them.

        Counters remain cumulative. New calls after clear start fresh work even
        when an older call for the same inputs is still running.
        """
        with self._lock:
            self._entries.clear()
            self._generation += 1

    def summary(self) -> dict:
        """Snapshot counters. waits counts callers sharing an in-flight request."""
        with self._lock:
            return dict(hits=self._hits, misses=self._misses, waits=self._waits,
                        evictions=self._evictions, expirations=self._expirations,
                        size=len(self._entries), in_flight=len(self._pending))


# =====================================================================
# Self-check (no network, no sleeping):  python api_layer.py
# =====================================================================
if __name__ == "__main__":
    class FakeClock:
        def __init__(self):
            self.t, self.sleeps = 0.0, []

        def now(self):
            return self.t

        def sleep(self, s):
            self.sleeps.append(s)
            self.t += s

    class HttpError(Exception):
        def __init__(self, msg, status_code=None):
            super().__init__(msg)
            self.status_code = status_code

    clock = FakeClock()
    api = ApiClient(limiter=RateLimiter(30, clock.now, clock.sleep), jitter=0, sleep=clock.sleep, verbose=False)

    # rate limiter: rpm=30 -> one request every 2 s
    stamps = []
    for _ in range(3):
        api.limiter.wait()
        stamps.append(clock.t)
    assert stamps == [0.0, 2.0, 4.0], stamps

    # retry: 429 twice, then success; every attempt goes through the limiter
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise HttpError("429 rate_limit_error", status_code=429)
        return "ok"

    assert api.call(flaky) == "ok" and len(calls) == 3
    # fatal and quota errors are not retried
    for message, expected in (("401 invalid x-api-key", FatalAPIError), ("429 quota exceeded PerDay", DailyQuotaError)):
        try:
            api.call(lambda m=message: (_ for _ in ()).throw(HttpError(m)))
        except expected:
            pass
        else:
            raise AssertionError(message)

    # batching: 5 items, batch_size 2 -> 3 requests, order preserved
    seen = []
    out = api.batch_call(list("abcde"), lambda xs: (seen.append(xs), [x.upper() for x in xs])[1], batch_size=2)
    assert [r.value for r in out] == list("ABCDE") and len(seen) == 3

    # concurrency: one bad item does not stop the rest
    res = api.run_concurrent(range(6), lambda n: 10 // (n - 3), workers=3, batch_size=3)
    assert [r.ok for r in res] == [True, True, True, False, True, True]
    print("all self-checks passed |", api.summary())
