"""serve/test_server.py - the max tokens budget over both APIs, against the mock engine (no GPU, no pack), plus
the resilience paths: every stream ends with a finish_reason (or an error event), a silent engine is aborted by
the watchdog and replaced, and reasoning_effort reaches the chat template.

    python -m unittest serve.test_server -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import CTX_SLACK, ByteTokenizer, MockEngine, Service, StrataEngine, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CTX = 4096
ANSWER = "x" * 2000                              # longer than the old 1024 fallback: one token per byte


class RecordingEngine(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_max_new = max_new
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class MaxTokens(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = RecordingEngine(tok, "</think>\n\n" + ANSWER, max_context=CTX)
        cls.svc = Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def call(self, api, text="hi", **budget):
        """-> (status, body, prompt tokens, completion tokens); `budget` is merged into the request as given."""
        msgs = [{"role": "user", "content": text}]
        if api == "openai":
            s, b = self.post("/v1/chat/completions", {"model": "m", "messages": msgs, **budget})
            u = b.get("usage", {})
            return s, b, u.get("prompt_tokens"), u.get("completion_tokens")
        s, b = self.post("/v1/messages", {"model": "m", "messages": msgs, **budget})
        u = b.get("usage", {})
        return s, b, u.get("input_tokens"), u.get("output_tokens")

    def test_unset_budget_is_the_rest_of_the_context(self):
        cases = {"openai": [{"max_tokens": -1}, {"max_tokens": 0}, {}, {"max_tokens": None},
                            {"max_completion_tokens": -1}, {"max_completion_tokens": None, "max_tokens": None}],
                 "anthropic": [{"max_tokens": -1}, {"max_tokens": 0}, {}, {"max_tokens": None}]}
        for api, budgets in cases.items():
            for budget in budgets:
                with self.subTest(api=api, budget=budget):
                    s, b, pt, ct = self.call(api, **budget)
                    self.assertEqual(s, 200, b)
                    self.assertEqual(self.engine.last_max_new, CTX - CTX_SLACK - pt)
                    self.assertGreater(ct, 1024)          # the whole answer, not cut at the old 1024 fallback

    def test_explicit_budget_is_honoured(self):
        for api, budget in [("openai", {"max_tokens": 50}), ("openai", {"max_completion_tokens": 50}),
                            ("openai", {"max_completion_tokens": 50, "max_tokens": 9}),
                            ("anthropic", {"max_tokens": 50}), ("openai", {"max_tokens": 1500}),
                            ("anthropic", {"max_tokens": 1500})]:
            with self.subTest(api=api, budget=budget):
                want = budget.get("max_completion_tokens") or budget["max_tokens"]
                s, b, _, ct = self.call(api, **budget)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, want)
                self.assertEqual(ct, want)

    def test_explicit_budget_over_the_context_is_clamped(self):
        for api in ("openai", "anthropic"):
            with self.subTest(api=api):
                s, b, pt, _ = self.call(api, max_tokens=CTX)
                self.assertEqual(s, 200, b)                       # never a 400: shortened to the room left
                self.assertEqual(self.engine.last_max_new, CTX - CTX_SLACK - pt)

    def test_unset_budget_with_a_near_full_prompt(self):
        _, _, pt0, _ = self.call("openai", max_tokens=1)
        overhead = pt0 - len("hi")                  # the template's tokens around the user text
        for api in ("openai", "anthropic"):
            _, _, pa, _ = self.call(api, max_tokens=1)
            over = pa - pt0                          # the Anthropic template may differ slightly
            with self.subTest(api=api, room=5):     # a few tokens left: the budget is exactly those
                text = "y" * (CTX - CTX_SLACK - overhead - over - 5)
                s, b, pt, ct = self.call(api, text=text, max_tokens=-1)
                self.assertEqual(s, 200, b)
                self.assertEqual(self.engine.last_max_new, 5)
                self.assertEqual(ct, 5)
            with self.subTest(api=api, room=0):     # nothing left: rejected, not truncated
                text = "y" * (CTX - CTX_SLACK - overhead - over)
                s, b, _, _ = self.call(api, text=text)
                self.assertEqual(s, 400, b)
                self.assertIn("no room to answer", b["error"]["message"])
            with self.subTest(api=api, room=0, explicit=1):   # ...same 400 with an explicit budget
                text = "y" * (CTX - CTX_SLACK - overhead - over)
                s, b, _, _ = self.call(api, text=text, max_tokens=1)
                self.assertEqual(s, 400, b)
                self.assertIn("no room to answer", b["error"]["message"])

    def test_debug_log_shows_the_resolved_budget(self):
        import contextlib
        import io
        os.environ["STRATA_DEBUG"] = "1"
        try:
            for api in ("openai", "anthropic"):
                with self.subTest(api=api):
                    out = io.StringIO()
                    with contextlib.redirect_stdout(out):
                        _, _, pt, _ = self.call(api, max_tokens=-1)
                    self.assertIn(f"max_new={CTX - CTX_SLACK - pt} ", out.getvalue())
        finally:
            del os.environ["STRATA_DEBUG"]


class WebApp(unittest.TestCase):
    """The web app (PR #22's dashboard idea, rebuilt): its page and files, and GET /metrics."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.svc = Service(RecordingEngine(tok, "</think>\n\nhello", max_context=CTX), tok,
                          ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def get(self, path, headers=None):
        req = urllib.request.Request(self.base + path, headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.headers.get("Content-Type", ""), r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.headers.get("Content-Type", ""), e.read()

    def test_page_and_files(self):
        code, ctype, body = self.get("/")
        self.assertEqual(code, 200)
        self.assertIn("text/html", ctype)
        self.assertIn(b"/web/app.js", body)
        for path, want in (("/web/app.js", "javascript"), ("/web/app.css", "text/css"), ("/web/tokens.css", "text/css"),
                           ("/web/components.css", "text/css"), ("/web/sprite.svg", "image/svg+xml")):
            with self.subTest(path=path):
                code, ctype, _ = self.get(path)
                self.assertEqual(code, 200)
                self.assertIn(want, ctype)

    def test_only_the_app_files_are_served(self):
        for path in ("/web/..%2Fserver.py", "/web/index.html", "/web/test.py", "/fonts/..%2F..%2Fsetup.py",
                     "/fonts/missing.woff2", "/fonts/x.ttf"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)

    def test_metrics(self):
        data = json.dumps({"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}).encode()
        urllib.request.urlopen(urllib.request.Request(self.base + "/v1/chat/completions", data=data,
                                                      headers={"Content-Type": "application/json"}), timeout=10).read()
        code, ctype, body = self.get("/metrics")
        self.assertEqual(code, 200)
        m = json.loads(body)
        for key in ("engine", "live", "requests", "hardware", "hardware_static", "history"):
            self.assertIn(key, m)
        self.assertEqual(m["engine"]["max_context"], CTX)
        self.assertEqual(m["live"]["state"], "idle")
        self.assertEqual(m["requests"][0]["output_tokens"], 5)

    def test_metrics_need_the_key_when_one_is_set(self):
        self.svc.api_key = "secret"
        try:
            self.assertEqual(self.get("/metrics")[0], 401)
            self.assertEqual(self.get("/metrics", {"Authorization": "Bearer secret"})[0], 200)
            self.assertEqual(self.get("/")[0], 200)                  # the page itself asks for the key
        finally:
            self.svc.api_key = ""


class RaisingEngine(MockEngine):
    """Streams a token or two, then fails the way the engine boundary can: an ERR (ValueError) or a fault."""

    def __init__(self, tok, error, after=2):
        super().__init__(tok, "</think>\n\nthe answer", max_context=CTX)
        self.error, self.after, self.emitted = error, after, 0

    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.last_prompt = list(ids)
        for _ in range(self.after):
            yield self.script[min(self.emitted, len(self.script) - 1)]
            self.emitted += 1
        raise self.error


class StreamTerminus(unittest.TestCase):
    """An engine fault must END the HTTP stream, not truncate it: clients that see a bare EOF report a protocol
    error (Pi: "Stream ended without finish_reason") and never learn what happened."""

    def setUp(self):
        self.tok = ByteTokenizer()

    def service_with(self, error):
        svc = Service(RaisingEngine(self.tok, error), self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        httpd = serve(svc, port=0)
        self.addCleanup(httpd.shutdown)
        return f"http://127.0.0.1:{httpd.server_address[1]}"

    @staticmethod
    def stream(base, path, body):
        req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode()

    @staticmethod
    def chunks(text):
        return [json.loads(line[6:]) for line in text.splitlines()
                if line.startswith("data: ") and line[6:] != "[DONE]"]

    def test_openai_stream_fault_ends_with_a_finish_reason(self):
        base = self.service_with(RuntimeError("boom"))
        text = self.stream(base, "/v1/chat/completions",
                           {"model": "m", "stream": True, "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]})
        last = self.chunks(text)[-1]
        self.assertEqual(last["choices"][0]["finish_reason"], "server_error")
        self.assertIn("boom", last["error"]["message"])
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))

    def test_openai_stream_err_line_ends_as_an_overflow(self):
        # an engine ERR about context size must finish_reason "no room to answer": pi-ai reports
        # "Provider finish_reason: <name>" and never reads a chunk's `error` field, so the overflow
        # class has to be encoded in the finish_reason itself for pi's compact-and-retry to fire
        base = self.service_with(ValueError("prompt (5 tokens) leaves no room to answer in the context (4096)"))
        text = self.stream(base, "/v1/chat/completions",
                           {"model": "m", "stream": True, "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]})
        last = self.chunks(text)[-1]
        self.assertEqual(last["choices"][0]["finish_reason"], "no room to answer")
        self.assertIn("no room to answer", last["error"]["message"])

    def test_stream_watchdog_fault_is_retryable_for_pi(self):
        # the watchdog's TimeoutError must finish_reason "server_error": pi's retryable pattern
        # `server.?error` matches the reported "Provider finish_reason: server_error" so the client
        # retries instead of dead-ending
        base = self.service_with(TimeoutError("the engine stopped responding (no output for 180 s)"))
        text = self.stream(base, "/v1/chat/completions",
                           {"model": "m", "stream": True, "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]})
        last = self.chunks(text)[-1]
        self.assertEqual(last["choices"][0]["finish_reason"], "server_error")
        self.assertIn("server error: ", last["error"]["message"])
        self.assertIn("no output for 180 s", last["error"]["message"])

    def test_anthropic_stream_fault_ends_with_an_error_event(self):
        base = self.service_with(RuntimeError("boom"))
        text = self.stream(base, "/v1/messages",
                           {"model": "m", "stream": True, "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]})
        self.assertIn("event: error", text)
        self.assertIn("event: message_stop", text)
        self.assertIn("boom", text)

    def test_non_stream_fault_is_a_500_with_the_message(self):
        for path in ("/v1/chat/completions", "/v1/messages"):
            with self.subTest(path=path):
                base = self.service_with(RuntimeError("boom"))
                body = {"model": "m", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]}
                req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                             headers={"Content-Type": "application/json"})
                with self.assertRaises(urllib.error.HTTPError) as cm:
                    urllib.request.urlopen(req, timeout=30)
                self.assertEqual(cm.exception.code, 500)
                self.assertIn("boom", json.dumps(json.loads(cm.exception.read())))


class Watchdog(unittest.TestCase):
    """Silence from the engine is a stalled request, not slowness: it is aborted (instead of waiting for hours)
    and the engine is replaced in the background."""

    def test_silence_aborts_the_request_and_restarts_the_engine(self):
        d = tempfile.mkdtemp(prefix="strata-watchdog-")
        exe = Path(d) / "silent-engine"
        exe.write_text("#!/usr/bin/env python3\n"
                       "import sys, time\n"
                       "print('INFO context=4096', flush=True)\n"
                       "print('READY 4096 stop', flush=True)\n"
                       "for line in sys.stdin:\n"
                       "    if line.startswith('QUIT'):\n"
                       "        break\n"
                       "    if line.startswith('GEN'):\n"
                       "        time.sleep(3600)\n")
        exe.chmod(0o755)
        eng = StrataEngine(str(exe), [], watchdog_s=1.0)
        self.addCleanup(eng.close)
        gen = eng.generate([1, 2, 3], 5, {}, threading.Event())
        started = time.time()
        with self.assertRaises(TimeoutError):
            for _ in range(200):
                next(gen)
        self.assertLess(time.time() - started, 40)
        deadline = time.time() + 15
        while not eng._ready.is_set() and time.time() < deadline:
            time.sleep(0.25)
        self.assertTrue(eng._ready.is_set(), "the engine was not restarted")


class RestartWindow(unittest.TestCase):
    """While the engine is being restarted (watchdog abort, engine death) the dead engine reads max_context 0:
    without a wait, every request in the window would 400 "leaves no room to answer in the context (0)" - a lie
    pi compacts on.  Requests must wait for the engine, then fail retryable if it never came back."""

    def service(self, pending_s):
        from types import SimpleNamespace
        eng = SimpleNamespace(_ready=threading.Event(), max_context=CTX, info={})
        if pending_s:
            threading.Timer(pending_s, eng._ready.set).start()
        self.tok = ByteTokenizer()
        return Service(eng, self.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def test_requests_wait_for_a_restarting_engine(self):
        svc = self.service(pending_s=1.0)
        ids, _, max_new = svc.prepare([{"role": "user", "content": "hi"}], [], {})
        self.assertEqual(max_new, CTX - CTX_SLACK - len(ids))   # normal budget once the engine is back

    def test_engine_that_never_returns_is_a_retryable_runtime_error(self):
        import serve.server as server
        svc = self.service(pending_s=None)
        old = server.READY_WAIT_S
        server.READY_WAIT_S = 1.0
        try:
            with self.assertRaises(RuntimeError) as cm:
                svc.prepare([{"role": "user", "content": "hi"}], [], {})
            self.assertIn("retrying is safe", str(cm.exception))
        finally:
            server.READY_WAIT_S = old


class DesyncGuard(unittest.TestCase):
    """A request that ends early (client disconnect, cancel, a drain that gave up) can leave the engine's DONE or
    its tokens unconsumed in the line queue.  The engine answers exactly one GEN at a time, so without a guard the
    NEXT request reads that stale output as its own: it returns instantly with 0 tokens and finish "length", which
    pi shows as "Response was truncated before completion" with nothing in it (reproduced live 2026-10-04).  The
    guard must detect the out-of-step queue, replace the engine, and serve the next request from a clean one."""

    @staticmethod
    def script() -> str:
        """A fake `--serve` engine; its last argv is a mode file read at startup: 'abandon' prints two tokens and a
        DONE for its first GEN (and never reads stdin again - the stranded-output shape), 'answer' answers every."""
        return ("#!/usr/bin/env python3\n"
                "import sys, time\n"
                "MODE = open(sys.argv[-1]).read().strip()\n"
                "print('INFO context=4096', flush=True)\n"
                "print('READY 4096 stop', flush=True)\n"
                "for line in sys.stdin:\n"
                "    if line.startswith('QUIT'):\n"
                "        break\n"
                "    if not line.startswith('GEN'):\n"
                "        continue\n"
                "    if MODE == 'abandon':\n"
                "        MODE = 'abandoned'\n"
                "        print('T 65', flush=True)\n"
                "        time.sleep(1.5)\n"
                "        print('T 66', flush=True)\n"
                "        print('DONE 2 3 1.0 40.0 cancel 0 0 0', flush=True)\n"
                "        time.sleep(3600)\n"
                "    else:\n"
                "        for t in (67, 68):\n"
                "            print('T %d' % t, flush=True)\n"
                "            time.sleep(0.02)\n"
                "        print('DONE 2 3 1.0 40.0 stop 0 0 0', flush=True)\n")

    def engine(self, mode: str) -> tuple[StrataEngine, Path]:
        d = tempfile.mkdtemp(prefix="strata-desync-")
        mode_file = Path(d) / "mode"
        mode_file.write_text(mode)
        exe = Path(d) / "fake-engine"
        exe.write_text(self.script())
        exe.chmod(0o755)
        eng = StrataEngine(str(exe), [str(mode_file)], watchdog_s=30.0)
        self.addCleanup(eng.close)
        return eng, mode_file

    def test_stale_output_is_never_served_to_the_next_request(self):
        import serve.server as server
        eng, mode_file = self.engine("abandon")
        old = server.DRAIN_S
        server.DRAIN_S = 0.5
        try:
            cancel = threading.Event()
            proc_before = eng.proc
            gen1 = eng.generate([1, 2, 3], 10, {}, cancel)
            self.assertEqual(next(gen1), 65)
            gen1.close()                     # consumer left early: the drain expires before the engine's DONE
            time.sleep(1.6)                  # the engine (ignoring STOP) finishes: its output is now stranded
            mode_file.write_text("answer")   # the replacement engine must be a clean one
            gen2 = eng.generate([1, 2, 3], 10, {}, cancel)
            got = [t for t in gen2 if t is not None]
            self.assertEqual(got, [67, 68], "the next request was served the stranded request's output")
            self.assertTrue(eng._ready.is_set())
        finally:
            server.DRAIN_S = old

    def test_preseeded_stale_done_triggers_replacement(self):
        eng, _ = self.engine("answer")
        cancel = threading.Event()
        proc_before = eng.proc
        eng.lines.put("DONE 9 9 0.0 0.0 cancel 0 0 0\n")   # a DONE stranded by a long-gone request
        gen = eng.generate([1, 2, 3], 10, {}, cancel)
        got = [t for t in gen if t is not None]
        self.assertEqual(got, [67, 68], "the next request was served the stranded DONE (phantom length)")
        self.assertIsNot(eng.proc, proc_before, "an out-of-step queue must replace the engine")

    def test_clean_finish_never_restarts_the_engine(self):
        eng, _ = self.engine("answer")
        cancel = threading.Event()
        proc_before = eng.proc
        for _ in range(2):
            got = [t for t in eng.generate([1, 2, 3], 10, {}, cancel) if t is not None]
            self.assertEqual(got, [67, 68])
        self.assertIs(eng.proc, proc_before, "a clean queue must not trigger an engine replacement")


class ThinkingControl(unittest.TestCase):
    """reasoning_effort reaches the chat template: none closes the think block, the levels map onto the model's."""

    @classmethod
    def setUpClass(cls):
        cls.tok = ByteTokenizer()
        cls.engine = RecordingEngine(cls.tok, "</think>\n\nhello", max_context=CTX)
        cls.svc = Service(cls.engine, cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.httpd = serve(cls.svc, port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()

    def prompt(self, **extra):
        body = {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}], **extra}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
        return self.tok.decode(self.engine.last_prompt)

    def test_none_disables_thinking(self):
        text = self.prompt(reasoning_effort="none")
        self.assertNotIn("Reasoning effort is set to xhigh", text)
        self.assertIn("<think>\n\n</think>", text)

    def test_default_is_the_template_default(self):
        self.assertIn("Reasoning effort is set to xhigh", self.prompt())

    def test_low_is_passed_through(self):
        self.assertIn("Reasoning effort is set to low", self.prompt(reasoning_effort="low"))


if __name__ == "__main__":
    unittest.main()
