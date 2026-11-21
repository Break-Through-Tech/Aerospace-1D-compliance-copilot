"""Request-cache behavior and integration with the teammate's API layer.

Run without third-party packages: python -m unittest discover -s tests -p test_api_layer.py -v
"""

import threading
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from unittest.mock import Mock, patch

from src.api_layer import ApiClient, DailyQuotaError, RateLimiter, RequestCache


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def now(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


class RequestCacheTests(unittest.TestCase):
    def test_repeated_request_skips_provider_and_rate_limiter(self):
        clock = FakeClock()
        api = ApiClient(limiter=RateLimiter(10, clock.now, clock.sleep), verbose=False)
        api.limiter.wait = Mock(wraps=api.limiter.wait)
        provider = Mock(return_value={"verdict": "Meets"})
        cache = RequestCache()
        judge = cache.wrap(lambda req, hits: api.call(provider, req, hits))

        first = judge("Document requirements", [("4.1.2", 0.9)])
        second = judge("Document requirements", [("4.1.2", 0.9)])

        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        provider.assert_called_once_with("Document requirements", [("4.1.2", 0.9)])
        api.limiter.wait.assert_called_once()
        self.assertEqual(clock.sleeps, [])
        self.assertEqual(api.summary()["requests"], 1)
        self.assertEqual(cache.summary()["hits"], 1)
        self.assertEqual(cache.summary()["misses"], 1)

    def test_equivalent_argument_spellings_and_mapping_order_hit(self):
        calls = Mock(return_value="ok")

        def fn(text, options=None, k=5):
            return calls(text, options, k)

        cached = RequestCache().wrap(fn)
        self.assertEqual(cached("test", {"a": 1, "b": 2}), "ok")
        self.assertEqual(cached(k=5, options={"b": 2, "a": 1}, text="test"), "ok")
        calls.assert_called_once()

    def test_changed_inputs_miss(self):
        provider = Mock(return_value="ok")
        cached = RequestCache().wrap(lambda req, hits: provider(req, hits))
        cached("one", [("clause", 0.9)])
        cached("two", [("clause", 0.9)])
        cached("one", [("changed clause", 0.9)])
        cached("one", [("clause", 0.8)])
        self.assertEqual(provider.call_count, 4)

    def test_input_types_cannot_collide(self):
        provider = Mock(return_value="ok")
        cached = RequestCache().wrap(lambda value: provider(value))
        for value in (True, 1, 1.0, "1", [1], (1,), {"1": 1}, {1: 1}):
            cached(value)
        self.assertEqual(provider.call_count, 8)

    def test_dataclasses_use_contents(self):
        @dataclass
        class Clause:
            text: str

        provider = Mock(return_value="ok")
        cached = RequestCache().wrap(lambda hits: provider(hits))
        cached([(Clause("document requirements"), 0.9)])
        cached([(Clause("document requirements"), 0.9)])
        cached([(Clause("approve requirements"), 0.9)])
        self.assertEqual(provider.call_count, 2)

    def test_wrappers_are_isolated_by_default(self):
        cache = RequestCache()
        first = cache.wrap(lambda req: "model A")
        second = cache.wrap(lambda req: "model B")
        self.assertEqual(first("same request"), "model A")
        self.assertEqual(second("same request"), "model B")

    def test_explicit_namespace_and_context_control_sharing(self):
        cache = RequestCache()
        provider = Mock(return_value="ok")

        def fn(req):
            return provider(req)

        for namespace, context in (
            ("judge", {"model": "a", "prompt": "v1"}),
            ("judge", {"prompt": "v1", "model": "a"}),
            ("judge", {"model": "b", "prompt": "v1"}),
            ("judge", {"model": "a", "prompt": "v2"}),
            ("embed", {"model": "a", "prompt": "v1"}),
        ):
            cache.wrap(fn, namespace=namespace, context=context)("test")
        self.assertEqual(provider.call_count, 4)

    def test_context_is_captured_when_wrapping(self):
        cache = RequestCache()
        context = {"model": "a"}
        cached = cache.wrap(lambda text: text, context=context)
        cached("test")
        context["model"] = "b"
        cached("test")
        self.assertEqual(cache.summary()["hits"], 1)

    def test_mutating_results_does_not_change_cached_value(self):
        original = {"missing": ["approval"]}
        cached = RequestCache().wrap(lambda: original)
        cached()["missing"].clear()
        original["missing"].append("tests")
        self.assertEqual(cached(), {"missing": ["approval"]})

    def test_falsey_results_are_cached(self):
        for value in (None, False, 0, "", [], {}):
            with self.subTest(value=value):
                provider = Mock(return_value=value)
                cached = RequestCache().wrap(lambda: provider())
                self.assertEqual(cached(), value)
                self.assertEqual(cached(), value)
                provider.assert_called_once()

    def test_expiration_starts_after_success_and_does_not_slide_on_hit(self):
        clock = FakeClock()
        cache = RequestCache(ttl=10, clock=clock.now)
        calls = []

        def slow():
            calls.append(1)
            clock.t += 20
            return "ok"

        cached = cache.wrap(slow)
        cached()  # finishes at 20, expires at 30
        clock.t = 29
        cached()
        clock.t = 30
        cached()
        self.assertEqual(len(calls), 2)
        self.assertEqual(cache.summary()["expirations"], 1)

    def test_lru_eviction(self):
        cache = RequestCache(maxsize=2)
        provider = Mock(side_effect=lambda value: value)
        cached = cache.wrap(lambda value: provider(value))
        for value in ("a", "b", "a", "c", "a", "b"):
            cached(value)
        self.assertEqual(provider.call_count, 4)
        self.assertEqual(cache.summary()["size"], 2)
        self.assertEqual(cache.summary()["evictions"], 2)

    def test_clear_invalidates_entries_and_preserves_counters(self):
        cache = RequestCache()
        provider = Mock(return_value="ok")
        cached = cache.wrap(lambda: provider())
        cached()
        cache.clear()
        self.assertEqual(cache.summary()["size"], 0)
        cached()
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(cache.summary()["misses"], 2)

    def test_errors_are_not_cached(self):
        for error in (ValueError("invalid output"), DailyQuotaError("daily quota")):
            with self.subTest(error=error):
                provider = Mock(side_effect=[error, "ok"])
                cache = RequestCache()
                cached = cache.wrap(lambda: provider())
                with self.assertRaises(type(error)):
                    cached()
                self.assertEqual(cached(), "ok")
                self.assertEqual(cached(), "ok")
                self.assertEqual(provider.call_count, 2)
                self.assertEqual(cache.summary()["in_flight"], 0)

    def test_transient_retry_result_is_cached(self):
        sleeps = []
        api = ApiClient(retries=3, sleep=sleeps.append, jitter=0, verbose=False)
        provider = Mock(side_effect=[TimeoutError("timeout"), "ok"])
        cached = RequestCache().wrap(lambda: api.call(provider))
        self.assertEqual(cached(), "ok")
        self.assertEqual(cached(), "ok")
        self.assertEqual(provider.call_count, 2)
        self.assertEqual(api.summary()["requests"], 2)
        self.assertEqual(api.summary()["retries"], 1)
        self.assertEqual(sleeps, [2.0])

    def test_concurrent_duplicates_share_success_or_failure(self):
        for fails in (False, True):
            with self.subTest(fails=fails):
                entered, release, waiting = threading.Event(), threading.Event(), threading.Event()

                class ObservedFuture(Future):
                    def result(self, timeout=None):
                        waiting.set()
                        return super().result(timeout)

                def provider(text):
                    entered.set()
                    if not release.wait(5):
                        raise AssertionError("provider was not released")
                    if fails:
                        raise ValueError("invalid output")
                    return {"text": text}

                provider_mock = Mock(side_effect=provider)
                cache = RequestCache()
                cached = cache.wrap(lambda text: provider_mock(text))
                with patch("src.api_layer.Future", ObservedFuture), ThreadPoolExecutor(2) as pool:
                    first = pool.submit(cached, "test")
                    try:
                        self.assertTrue(entered.wait(5))
                        second = pool.submit(cached, "test")
                        self.assertTrue(waiting.wait(5))
                    finally:
                        release.set()
                    if fails:
                        for future in (first, second):
                            with self.assertRaisesRegex(ValueError, "invalid output"):
                                future.result(timeout=5)
                        self.assertEqual(cache.summary()["size"], 0)
                    else:
                        self.assertEqual(first.result(timeout=5), second.result(timeout=5))
                        self.assertIsNot(first.result(), second.result())
                provider_mock.assert_called_once()
                self.assertEqual(cache.summary()["waits"], 1)
                self.assertEqual(cache.summary()["in_flight"], 0)
                if fails:
                    with self.assertRaises(ValueError):
                        cached("test")
                    self.assertEqual(provider_mock.call_count, 2)

    def test_distinct_keys_run_in_parallel(self):
        barrier = threading.Barrier(2, timeout=5)

        def provider(text):
            barrier.wait()
            return text

        cached = RequestCache().wrap(provider)
        with ThreadPoolExecutor(2) as pool:
            self.assertEqual(list(pool.map(cached, ["a", "b"])), ["a", "b"])

    def test_clear_during_request_prevents_repopulation(self):
        entered, release = threading.Event(), threading.Event()
        calls = []

        def provider():
            calls.append(1)
            if len(calls) == 1:
                entered.set()
                if not release.wait(5):
                    raise AssertionError("provider was not released")
                return "old"
            return "new"

        cache = RequestCache()
        cached = cache.wrap(provider)
        with ThreadPoolExecutor(1) as pool:
            old = pool.submit(cached)
            try:
                self.assertTrue(entered.wait(5))
                cache.clear()
                self.assertEqual(cached(), "new")
            finally:
                release.set()
            self.assertEqual(old.result(timeout=5), "old")
        self.assertEqual(cached(), "new")
        self.assertEqual(len(calls), 2)

    def test_unsupported_inputs_fail_before_provider_call(self):
        provider = Mock()
        cached = RequestCache().wrap(lambda value: provider(value))
        for value in (object(), float("nan"), float("inf"), {1, 2}):
            with self.assertRaises(TypeError):
                cached(value)
        provider.assert_not_called()

    def test_uncopyable_result_does_not_leave_pending_request(self):
        cache = RequestCache()
        cached = cache.wrap(lambda: threading.Lock())
        for _ in range(2):
            with self.assertRaises(TypeError):
                cached()
        self.assertEqual(cache.summary()["in_flight"], 0)
        self.assertEqual(cache.summary()["size"], 0)

    def test_invalid_configuration_and_arguments(self):
        for maxsize in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                RequestCache(maxsize=maxsize)
        for ttl in (0, -1, True, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                RequestCache(ttl=ttl)
        for namespace in ("", "  ", 1):
            with self.assertRaises(ValueError):
                RequestCache().wrap(lambda: None, namespace=namespace)
        cached = RequestCache().wrap(lambda text: text)
        with self.assertRaises(TypeError):
            cached()


if __name__ == "__main__":
    unittest.main()
