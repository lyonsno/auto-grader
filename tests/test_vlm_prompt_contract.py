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
            3900,
            "system prompt should stay reasonably compact even after the positive-sweep graft on top of the integration surface's stable grading contracts",
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
        self.assertEqual(
            version,
            "2026-04-11-positive-sweep-v1",
            "integration tip should expose the positive-sweep prompt identity",
        )

    def test_system_prompt_uses_explicit_rescue_credit_language(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertNotIn(
            "Be charitable.",
            prompt,
            "generic charity language is too vague for the grading target here",
        )
        self.assertNotIn(
            "erring on the side of generosity",
            prompt,
            "prompt should ask for the actual rescue objective, not vague generosity phrasing",
        )
        self.assertIn(
            "Award the highest score justified by the student's written work under the rubric.",
            prompt,
            "prompt should optimize for the highest justified score, not generic leniency",
        )
        self.assertIn(
            "Actively rescue as much lawful partial credit as possible",
            prompt,
            "prompt should explicitly say to rescue rubric-grounded partial credit",
        )

    def test_system_prompt_splits_notation_charity_from_scoring_charity(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "Be charitable toward handwriting and notation: if a student's marks admit a reasonable reading as correct, read them that way.",
            prompt,
            "prompt should separate perception charity from score charity",
        )
        self.assertIn(
            'Be strict toward errors you see. An error you notice is an error you grade, even if the student "demonstrated the core concept"',
            prompt,
            "prompt should explicitly block the fr-12b-style charity rationalization",
        )
        self.assertIn(
            "abandoning the rubric, not charity.",
            prompt,
            "prompt should keep the named anti-charity failure mode in positive framing",
        )

    def test_system_prompt_prefers_lawful_full_credit_and_equivalent_units(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "If the student's work supports a lawful full-credit interpretation, take it and stop.",
            prompt,
            "prompt should prefer a supportable full-credit reading over continued nitpicking",
        )
        self.assertIn(
            "Treat mL and cm³ as equivalent unless the question explicitly tests form.",
            prompt,
            "prompt should treat mL and cm³ as equivalent when the form itself is not being tested",
        )
        self.assertIn(
            "small arithmetic, truncation, or rounding",
            prompt,
            "near-correct numerics should not lose points just for tiny arithmetic or rounding slips",
        )
        self.assertIn(
            "award full credit unless exact rounding or significant figures are being tested",
            prompt,
            "numeric rescue should still respect questions that explicitly test rounding or sig figs",
        )

    def test_system_prompt_defaults_dependency_to_none_unless_clear(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            'Use upstream_dependency = "none" unless carry-forward is clear.',
            prompt,
            "dependency handling should default to none unless the carry-forward is clear",
        )

    def test_system_prompt_limits_charitable_reread_loops(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "Grade what is written, not a more favorable answer you can imagine.",
            prompt,
            "prompt should stop the model from rescuing borderline OCR reads through speculation",
        )
        self.assertIn(
            "If two readings are plausible and neither is clearly better supported, choose the best-supported reading and move on.",
            prompt,
            "prompt should narrow ambiguity handling so the model does not loop on rereads",
        )

    def test_system_prompt_tells_easy_wrong_form_items_to_stop_early(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "If the requested answer form is plainly missing, stop and score only what is on the page.",
            prompt,
            "easy wrong-form items should not invite long re-litigation after the missing form is already clear",
        )

    def test_system_prompt_uses_bounded_effort_handoff_for_hard_ambiguity(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "After one careful pass, if ambiguity still affects the score, choose the best-supported reading, say in model_reasoning that human review is warranted, lower model_confidence, and stop.",
            prompt,
            "hard ambiguous items should hand off cleanly once bounded effort is exhausted",
        )

    def test_system_prompt_distinguishes_score_basis_from_model_reasoning(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "score_basis = short literal basis for the awarded score",
            prompt,
            "prompt should define score_basis as the direct basis for the points awarded",
        )
        self.assertIn(
            "model_reasoning = broader reasoning only",
            prompt,
            "prompt should reserve model_reasoning for broader interpretive reasoning instead of score restatement",
        )
        self.assertIn(
            "Do not restate score_basis in model_reasoning.",
            prompt,
            "prompt should explicitly prevent overlap between score_basis and model_reasoning",
        )
        self.assertIn(
            '"score_basis": <string>',
            prompt,
            "the JSON schema should persist score_basis as a first-class field",
        )

    def test_system_prompt_declares_obvious_correctness_buckets(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "Use is_obviously_fully_correct = true only for clearly correct answers needing no human rescue.",
            prompt,
            "prompt should expose a high-trust obvious-full-credit bucket",
        )
        self.assertIn(
            "Use is_obviously_wrong = true only for clearly wrong answers with no lawful rescue path.",
            prompt,
            "prompt should expose a high-trust obvious-wrong bucket",
        )
        self.assertIn(
            '"is_obviously_fully_correct": <true | false | null>',
            prompt,
            "the JSON schema should persist the obvious-full-credit bucket",
        )
        self.assertIn(
            '"is_obviously_wrong": <true | false | null>',
            prompt,
            "the JSON schema should persist the obvious-wrong bucket",
        )

    def test_system_prompt_keeps_obvious_wrong_out_of_partial_credit_cases(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "Do not use is_obviously_wrong = true if any lawful partial-credit path remains.",
            prompt,
            "obvious-wrong should be reserved for true zero-credit cases, not harsh partial-credit judgments",
        )

    def test_system_prompt_blocks_rescue_credit_on_answered_form_failures(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "When the requested form is the thing being graded, do not award rescue credit for nearby ingredients unless the rubric explicitly does so.",
            prompt,
            "answered-form failures should not pick up rescue points just for mentioning nearby chemistry",
        )

    def test_system_prompt_explicitly_rescues_partial_credit_on_lewis_structures(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "On Lewis-structure questions, rescue partial credit for correct connectivity",
            prompt,
            "Lewis-structure grading should rescue meaningful structural progress",
        )
        self.assertIn(
            "even if octets, formal charges, or resonance are incomplete.",
            prompt,
            "Lewis-structure grading should preserve partial credit for meaningful correct substructure",
        )

    def test_system_prompt_preserves_setup_credit_when_relation_is_right(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "Right relation but later execution or unit miss: preserve nonzero setup credit unless the setup itself is wrong.",
            prompt,
            "setup-credit numerics like fr-10a should not collapse toward zero once the governing relation is correct",
        )

    def test_system_prompt_distinguishes_wrong_concept_from_wrong_execution(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "Wrong-concept vs wrong-execution: preserve method credit for right approach with bad arithmetic or units, but not for a wrong approach that only shares surface symbols with the right one.",
            prompt,
            "prompt should block method credit for wrong-approach answers that merely look nearby",
        )
        self.assertIn(
            "If the student's approach would still be wrong with perfect execution, do not award method credit.",
            prompt,
            "prompt should distinguish bad execution from a fundamentally wrong method",
        )

    def test_system_prompt_rescues_lewis_partial_credit_from_zero_when_structure_basis_is_present(self):
        from auto_grader import vlm_inference

        prompt = vlm_inference._SYSTEM_PROMPT
        self.assertIn(
            "Do not collapse a Lewis-structure answer to zero when connectivity or the valence-electron basis is clearly right and only bonding or octet completion is wrong.",
            prompt,
            "Lewis partial-credit cases like 34-blue/fr-12a should stay out of the zero bucket when the structural basis is present",
        )


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
