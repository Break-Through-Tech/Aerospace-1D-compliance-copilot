"""Offline request-cache demo: python3 -m scripts.demo_request_cache.

Uses a simulated provider and clock: no API key, model, network, or real sleeps.
"""

import json

from src.api_layer import ApiClient, RateLimiter, RequestCache


class DemoClock:
    def __init__(self):
        self.seconds = 0.0

    def now(self):
        return self.seconds

    def sleep(self, seconds):
        self.seconds += seconds


class CountingLimiter(RateLimiter):
    def __init__(self, clock):
        super().__init__(rpm=10, clock=clock.now, sleep=clock.sleep)
        self.entries = 0

    def wait(self):
        self.entries += 1
        return super().wait()


class DemoProvider:
    def __init__(self):
        self.calls = 0

    def respond(self, requirement, hits):
        self.calls += 1
        return {"provider_call": self.calls, "requirement": requirement, "context": hits}


def main():
    clock = DemoClock()
    limiter = CountingLimiter(clock)
    api = ApiClient(limiter=limiter, verbose=False)
    provider = DemoProvider()
    cache = RequestCache(maxsize=128, ttl=300, clock=clock.now)
    cached_request = cache.wrap(
        lambda requirement, hits: api.call(provider.respond, requirement, hits),
        context={"provider": "offline-demo", "model": "demo-v1", "prompt": "v1"},
    )
    original_hits = [{"clause_id": "4.1.2", "text": "Document requirements."}]
    changed_hits = [{"clause_id": "4.1.2", "text": "Document and approve requirements."}]
    requests = [
        ("Document requirements", original_hits),
        ("Document requirements", original_hits),
        ("Test requirements", original_hits),
        ("Document requirements", original_hits),
        ("Document requirements", changed_hits),
        ("Document requirements", original_hits),
    ]

    print("Offline demo: simulated provider and clock; no real requests or sleeps.\n")
    for number, (requirement, hits) in enumerate(requests, 1):
        calls_before, slots_before = provider.calls, limiter.entries
        response = cached_request(requirement, hits)
        reused = provider.calls == calls_before
        if reused and limiter.entries != slots_before:
            raise RuntimeError("A cache hit entered the rate limiter")
        status = "HIT " if reused else "MISS"
        print(f"{number}. {status} | {requirement} | response from provider call #{response['provider_call']}")

    stats = cache.summary()
    if (provider.calls, limiter.entries, api.stats.requests, stats["hits"], stats["misses"]) != (3, 3, 3, 3, 3):
        raise RuntimeError("Unexpected demo counts: repeated requests were not reused correctly")
    print("\n" + json.dumps({
        "logical_requests": len(requests),
        "provider_calls": provider.calls,
        "rate_limiter_entries": limiter.entries,
        "cache": stats,
        "api": api.summary(),
    }, indent=2))
    print("\nPASS: 6 requests, 3 provider calls, 3 cache hits; hits bypassed the rate limiter.")


if __name__ == "__main__":
    main()
