# Request cache usage

`src/api_layer.py` combines Venya's rate limiting, retries and batching with
Aaliyan's request cache in section 4. It uses only the Python standard library
and accepts a synchronous callable for any provider.

This module is available for callers to opt into. The existing repository
pipelines do not automatically use it. Their persistent audit caches remain
available for resuming runs; this request cache keeps results in memory for the
lifetime of the cache instance. Restarting Python starts with an empty cache.

## Run the offline demo

From the repository root, using Python 3.10 or newer:

```bash
python3 -m scripts.demo_request_cache
```

The demo uses a simulated provider and clock. No dependencies, API keys, model
downloads, network access or real sleeps are needed. It makes six logical
requests, including duplicates, a different requirement, and changed clause text.
Expected output includes:

```text
1. MISS | Document requirements | response from provider call #1
2. HIT  | Document requirements | response from provider call #1
3. MISS | Test requirements | response from provider call #2
4. HIT  | Document requirements | response from provider call #1
5. MISS | Document requirements | response from provider call #3
6. HIT  | Document requirements | response from provider call #1
```

The final summary reports three provider calls, three rate-limiter entries,
three hits and three misses. The demo exits with an error if a hit enters the rate
limiter or the counts differ. Reported wait times come from the simulated clock.

## Wrap a request

Create the wrapper once and reuse it. Given your provider's `judge` object:

```python
from src.api_layer import ApiClient, RequestCache

api = ApiClient(rpm=10, verbose=False)
cache = RequestCache(maxsize=1024, ttl=300)

cached_judge = cache.wrap(
    lambda requirement, hits: api.call(judge.judge, requirement, hits),
    context={"provider": "ollama", "model": "your-fixed-model-version", "prompt": "v1"},
)

first = cached_judge(requirement_text, retrieved_hits)
second = cached_judge(requirement_text, retrieved_hits)
print(cache.summary())
print(api.summary())
```

The cache surrounds `api.call`: a hit returns before rate limiting or retries.
Calling `api.call(cached_judge, ...)` would still enter the limiter on each hit.
Recreating `cache.wrap(...)` on every request also prevents reuse with the
default private wrapper scope.

Only successful returns are stored. If a provider returns error objects instead
of raising, validate inside the wrapped function and raise on invalid results.
Include response validation in the wrapper so invalid responses are not reused.
Exceptions, including exhausted retries and quota errors, are propagated and
never cached; a later call can retry the same inputs.

## Keys and configuration

Keys include all bound arguments, their types, and the supplied `context`.
Dictionary insertion order, keyword argument order, explicit defaults, and
equivalent positional/keyword spelling do not change the key. Changes to
requirement text, clause text, clause order, or retrieval scores produce misses.
List and tuple inputs remain distinct, as do booleans, integers and floats.

Supported inputs are `None`, strings, booleans, integers, finite floats, lists,
tuples, dictionaries, enums, dataclass instances, and objects exposing Pydantic
v2's `model_dump`. Nested values must also be supported. Convert arrays, sets,
streams and other unsupported inputs to these types first. Results must support
`copy.deepcopy`; callers receive separate copies so mutations do not alter the
cached result. Compatibility with repository verdict models has not been tested
as part of this change.

Every wrapper has a private scope by default. To share entries between separately
created wrappers, supply the same explicit `namespace` and `context`, and use
equivalent operations with matching parameter names and defaults:

```python
cached_judge = cache.wrap(
    request_fn,
    namespace="compliance-judge-v1",
    context={
        "provider": "ollama",
        "endpoint": "http://localhost:11434",
        "model": "your-fixed-model-version",
        "prompt": "v1",
        "temperature": 0,
        "validation": "v1",
    },
)
```

Context is captured when wrapping. The cache cannot discover changes to objects
closed over by `request_fn`, including model settings, prompts or remote model
contents. Recreate the wrapper with updated context, or call `cache.clear()`,
when those settings change. A namespace alone does not identify model settings.

## Capacity, expiration and concurrency

`maxsize` is the maximum number of completed results retained. The least recently
used entry is evicted when a new result exceeds capacity. `ttl` is measured in
seconds from successful completion; hits do not extend it. The default `ttl=None`
keeps results until eviction or clearing. Expired entries are removed when their
key is requested or displaced by capacity eviction, so `size` may include entries
whose TTL has elapsed.

Concurrent identical requests through the same scope share one in-flight call.
Other keys can execute in parallel. Waiting callers receive the result or error
from that shared call. The cache is shared across threads using the same instance;
separate processes or cache instances have separate contents. In-flight requests
are not counted toward `maxsize`; use a worker limit to bound concurrency.

For concurrent requests, submit the cached wrapper directly:

```python
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(lambda request: cached_judge(*request), requests))
```

`ApiClient.run_concurrent` and `ApiClient.batch_call` call `api.call` internally.
Passing a cached function to those helpers would still enter their outer rate
limiter. Use the executor pattern above when hits must bypass the limiter.

`cache.clear()` invalidates stored results and preserves cumulative counters.
Existing in-flight callers finish, but their old results do not refill the
cleared cache; new requests start fresh work.

## Inspect and test

`cache.summary()` reports:

| Field | Meaning |
| --- | --- |
| `hits` | Calls served from completed, unexpired entries |
| `misses` | Calls that started work, including calls that later failed |
| `waits` | Calls that shared an in-flight request |
| `evictions` | Entries removed to enforce capacity |
| `expirations` | Expired entries removed on lookup |
| `size` | Completed entries currently retained |
| `in_flight` | Requests currently executing |

`api.summary()["requests"]` counts provider attempts, including retries, so it
can exceed cache misses. Hits and waiting callers do not add provider attempts.

Run the focused tests and the teammate's original API self-check without
third-party dependencies:

```bash
python3 -m unittest discover -s tests -p test_api_layer.py -v
python3 -m src.api_layer
```

These cache tests use simulated providers and clocks, with no live-model calls.
