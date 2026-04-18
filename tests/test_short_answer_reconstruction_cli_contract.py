from __future__ import annotations

import os
from pathlib import Path
import json
import subprocess
import tempfile
import unittest


def _asset_root() -> Path:
    configured = os.environ.get("AUTO_GRADER_ASSETS_DIR")
    if configured:
        return Path(configured) / "exams"
    return Path.home() / "dev" / "auto-grader-assets" / "exams"


_ASSET_ROOT = _asset_root()
_QUIZ_A = _ASSET_ROOT / "260326_Quiz _5 A.pdf"
_QUIZ_B = _ASSET_ROOT / "260326_Quiz _5 B.pdf"


def _load_writer(test_case: unittest.TestCase):
    try:
        from auto_grader.quiz5_short_answer_reconstruction import (
            write_reconstructed_short_answer_quiz_family,
        )
    except ModuleNotFoundError:
        test_case.fail(
            "Keep `auto_grader.quiz5_short_answer_reconstruction` importable so the "
            "short-answer reconstruction lane has a stable repo-local surface."
        )
    except ImportError:
        test_case.fail(
            "Export `write_reconstructed_short_answer_quiz_family(...)` so the "
            "Quiz #5 reconstruction can be driven as a real file-in/file-out tool."
        )
    return write_reconstructed_short_answer_quiz_family


@unittest.skipUnless(_QUIZ_A.exists() and _QUIZ_B.exists(), "Quiz #5 legacy PDFs are required for this contract")
class ShortAnswerReconstructionCliContractTests(unittest.TestCase):
    def test_writer_emits_family_json_bundle_for_real_quiz_pair(self) -> None:
        write_reconstructed_short_answer_quiz_family = _load_writer(self)

        with tempfile.TemporaryDirectory(prefix="short-answer-reconstruction-") as tempdir:
            result = write_reconstructed_short_answer_quiz_family(
                pdf_paths=[_QUIZ_A, _QUIZ_B],
                output_dir=tempdir,
            )

            family_path = Path(result["family_json_path"])
            self.assertTrue(family_path.exists())
            self.assertEqual(family_path.name, "short-answer-quiz-family.json")

            family = json.loads(family_path.read_text(encoding="utf-8"))
            self.assertEqual(family["slug"], "chm142-quiz-5")
            self.assertEqual(set(family["variants"]), {"A", "B"})

    def test_cli_script_writes_family_json_and_prints_result_paths(self) -> None:
        script_path = Path("scripts/reconstruct_quiz5_short_answer.py")
        self.assertTrue(
            script_path.exists(),
            "Add `scripts/reconstruct_quiz5_short_answer.py` so the Quiz #5 reconstruction "
            "flow is available as a CLI tool instead of only a Python import seam.",
        )

        with tempfile.TemporaryDirectory(prefix="short-answer-reconstruction-cli-") as tempdir:
            proc = subprocess.run(
                [
                    str(Path(".venv/bin/python")),
                    str(script_path),
                    "--output-dir",
                    tempdir,
                    "--pdf",
                    str(_QUIZ_A),
                    "--pdf",
                    str(_QUIZ_B),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(
                proc.returncode,
                0,
                proc.stderr,
            )

            result = json.loads(proc.stdout)
            family_path = Path(result["family_json_path"])
            self.assertTrue(family_path.exists())
            self.assertEqual(family_path.name, "short-answer-quiz-family.json")

    def test_cli_can_emit_reviewable_generated_variant_c(self) -> None:
        script_path = Path("scripts/reconstruct_quiz5_short_answer.py")

        with tempfile.TemporaryDirectory(prefix="short-answer-reconstruction-generate-") as tempdir:
            proc = subprocess.run(
                [
                    str(Path(".venv/bin/python")),
                    str(script_path),
                    "--output-dir",
                    tempdir,
                    "--pdf",
                    str(_QUIZ_A),
                    "--pdf",
                    str(_QUIZ_B),
                    "--generate-variant",
                    "C",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(proc.returncode, 0, proc.stderr)

            result = json.loads(proc.stdout)
            generated_path = Path(result["generated_variant_json_path"])
            self.assertTrue(generated_path.exists())
            self.assertEqual(generated_path.name, "short-answer-quiz-variant-C.json")

            generated = json.loads(generated_path.read_text(encoding="utf-8"))
            self.assertEqual(generated["variant_id"], "C")
            self.assertIn("question_prompts", generated)

    def test_cli_reports_usage_when_required_arguments_are_missing(self) -> None:
        script_path = Path("scripts/reconstruct_quiz5_short_answer.py")
        proc = subprocess.run(
            [str(Path(".venv/bin/python")), str(script_path)],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("usage:", proc.stderr)


if __name__ == "__main__":
    unittest.main()
