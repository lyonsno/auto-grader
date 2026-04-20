from __future__ import annotations

import http.server
import threading
import unittest


# ---------------------------------------------------------------------------
# Stub HTTP server helpers
# ---------------------------------------------------------------------------

class _StubHandler(http.server.BaseHTTPRequestHandler):
    """Returns a canned 500 response with a known body for every POST."""

    response_body: bytes = b""  # set by the test before starting

    def do_POST(self):
        # Drain the request body so the client doesn't get a broken pipe.
        length = int(self.headers.get("Content-Length", 0))
        if length:
            self.rfile.read(length)
        self.send_response(500)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(self.response_body)))
        self.end_headers()
        self.wfile.write(self.response_body)

    def log_message(self, *args):
        pass  # suppress test noise


class _StubServer:
    """Context manager that starts a stub HTTP server on an ephemeral port."""

    def __init__(self, body: bytes):
        self._body = body
        self._server = None

    def __enter__(self):
        handler = type(
            "_Handler",
            (_StubHandler,),
            {"response_body": self._body},
        )
        self._server = http.server.HTTPServer(("127.0.0.1", 0), handler)
        port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return f"http://127.0.0.1:{port}"

    def __exit__(self, *_):
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if hasattr(self, "_thread"):
            self._thread.join(timeout=1)


# ---------------------------------------------------------------------------
# Prompt contract tests
# ---------------------------------------------------------------------------


class GradingPromptContract(unittest.TestCase):
    def test_system_prompt_is_concise_and_has_no_worked_example(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertNotIn(
            "Worked example:",
            prompt,
            "system prompt should state the rule directly, not sermonize with a worked example",
        )
        self.assertLess(
            len(prompt),
            4500,
            "the Qwen 3.6 structured scaffold may be longer than the earlier "
            "compact prompt, but it still needs a real ceiling so we don't "
            "silently drift into runaway policy prose",
        )

    def test_system_prompt_states_each_major_rule_once(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT.lower()
        self.assertEqual(
            prompt.count("answered-form rule"),
            1,
            "answered-form rule should be introduced once, clearly",
        )
        self.assertLessEqual(
            prompt.count("upstream"),
            4,
            "upstream-dependency handling should be clear without being repeated to death",
        )
        self.assertLessEqual(
            prompt.count("consistency"),
            2,
            "consistency rule should be stated cleanly, not reiterated in multiple phrasings",
        )

    def test_system_prompt_uses_transcribe_method_score_scaffold(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn("1. TRANSCRIBE:", prompt)
        self.assertIn("2. IDENTIFY METHOD:", prompt)
        self.assertIn("3. SCORE:", prompt)
        self.assertIn(
            "Structure your analysis in three steps:",
            prompt,
            "the Qwen 3.6 smoke surface should carry the structured scaffold "
            "that prevents productive-but-unbounded exploration from spiraling",
        )

    def test_prompt_contract_publishes_version_and_hash(self):
        from auto_grader import vlm_inference

        version = getattr(vlm_inference, "GRADING_PROMPT_VERSION", "")
        self.assertRegex(
            version,
            r"^\d{4}-\d{2}-\d{2}-",
            "grading prompt should declare a human-readable version string in code",
        )

        metadata_fn = getattr(vlm_inference, "grading_prompt_metadata", None)
        self.assertTrue(
            callable(metadata_fn),
            "vlm prompt contract should publish grading_prompt_metadata() for run manifests",
        )
        metadata = metadata_fn()
        self.assertEqual(metadata["version"], version)
        self.assertRegex(metadata["content_hash"], r"^[0-9a-f]{64}$")


class HTTPErrorBodyContract(unittest.TestCase):
    """Operation Body Bag (err-capture lane).

    grade_single_item must preserve the server's 500 response body in the
    raised TimeoutError so operators can diagnose model-specific crashes.
    Before the fix, the body was silently discarded and the exception only
    carried the generic "HTTP Error 500: Internal Server Error" phrase.
    """

    def test_http_500_body_appears_in_timeout_error(self):
        """Stub server returns 500 with a known traceback body.
        grade_single_item must raise TimeoutError whose message contains
        the body — not just the generic HTTP status phrase.
        """
        from auto_grader.eval_harness import EvalItem
        from auto_grader.vlm_inference import ServerConfig, grade_single_item

        known_body = (
            b"Traceback (most recent call last):\n"
            b"  File 'mlx_server.py', line 42, in chat_completions\n"
            b"    raise RuntimeError('chat template not found for Step3-VL-10B')\n"
            b"RuntimeError: chat template not found for Step3-VL-10B\n"
        )

        item = EvalItem(
            exam_id="15-blue",
            question_id="fr-10b",
            answer_type="numeric",
            page=1,
            professor_score=0.0,
            max_points=3.0,
            professor_mark="x",
            student_answer="",
            notes="",
        )
        fake_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64  # minimal placeholder

        with _StubServer(known_body) as base_url:
            config = ServerConfig(base_url=base_url, api_key="test")
            with self.assertRaises(TimeoutError) as ctx:
                grade_single_item(item, fake_image, config)

        msg = str(ctx.exception)
        self.assertIn(
            "chat template not found for Step3-VL-10B",
            msg,
            f"Expected server traceback body in TimeoutError message, got: {msg!r}",
        )


if __name__ == "__main__":
    unittest.main()
