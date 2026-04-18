"""Contract tests for canonical short-answer quiz-family reconstruction."""

from __future__ import annotations

import os
from pathlib import Path
import unittest


def _asset_root() -> Path:
    configured = os.environ.get("AUTO_GRADER_ASSETS_DIR")
    if configured:
        return Path(configured) / "exams"
    return Path.home() / "dev" / "auto-grader-assets" / "exams"


_ASSET_ROOT = _asset_root()
_QUIZ_A = _ASSET_ROOT / "260326_Quiz _5 A.pdf"
_QUIZ_B = _ASSET_ROOT / "260326_Quiz _5 B.pdf"


@unittest.skipUnless(_QUIZ_A.exists() and _QUIZ_B.exists(), "Quiz #5 legacy PDFs are required for this contract")
class TestShortAnswerQuizFamilyReconstruction(unittest.TestCase):
    """The Quiz #5 A/B legacy forms should collapse into one canonical family."""

    def _reconstruct(self):
        from auto_grader.quiz5_short_answer_reconstruction import (
            reconstruct_short_answer_quiz_family,
        )

        return reconstruct_short_answer_quiz_family([_QUIZ_A, _QUIZ_B])

    def test_module_and_entrypoint_exist(self) -> None:
        from auto_grader.quiz5_short_answer_reconstruction import (
            reconstruct_short_answer_quiz_family,
        )

        self.assertTrue(callable(reconstruct_short_answer_quiz_family))

    def test_reconstruction_returns_one_canonical_template_for_a_and_b(self) -> None:
        family = self._reconstruct()

        self.assertEqual(family["slug"], "chm142-quiz-5")
        self.assertEqual(family["title"], "CHM 142 Quiz #5")
        self.assertEqual(set(family["variants"]), {"A", "B"})

        template = family["template"]
        self.assertEqual(template["slug"], "chm142-quiz-5")
        self.assertEqual(template["title"], "CHM 142 Quiz #5")
        self.assertEqual(len(template["sections"]), 1)
        self.assertEqual(template["sections"][0]["id"], "short-answer")

    def test_reconstruction_preserves_shared_question_and_box_structure(self) -> None:
        family = self._reconstruct()
        questions = family["template"]["sections"][0]["questions"]

        self.assertEqual([question["id"] for question in questions], ["q1", "q2", "q3", "q4", "q5", "q6"])

        q1 = questions[0]
        self.assertEqual([part["id"] for part in q1["parts"]], ["q1a", "q1b"])
        self.assertEqual([part["response_box_label"] for part in q1["parts"]], ["1a.", "1b."])

        q6 = questions[-1]
        self.assertEqual([part["id"] for part in q6["parts"]], ["q6-ch4", "q6-ccl4", "q6-ch2cl2"])
        self.assertEqual(
            [part["response_box_label"] for part in q6["parts"]],
            ["6.[CH4]=", "6.[CCl4]=", "6.[CH2Cl2]="],
        )

    def test_reconstruction_captures_variant_specific_substitutions_without_forking_questions(self) -> None:
        family = self._reconstruct()
        variants = family["variants"]

        self.assertEqual(
            variants["A"]["variables"],
            {
                "acid_molarity": 1.68e-2,
                "acid_species": "hydroiodic acid",
                "base_molarity": 0.0589,
                "bronsted_acid": "isobutyric acid (CH3)2COOH",
                "bronsted_base": "dimethylamine (CH3)2NH",
                "kc_q5": 0.0029,
                "kc_q6": 0.0952,
                "target_ph": 2.00,
            },
        )
        self.assertEqual(
            variants["B"]["variables"],
            {
                "acid_molarity": 2.68e-3,
                "acid_species": "hydrobromic acid",
                "base_molarity": 0.00890,
                "bronsted_acid": "butanoic acid C3H7COOH",
                "bronsted_base": "pyridine C5H5N",
                "kc_q5": 0.0049,
                "kc_q6": 0.0476,
                "target_ph": 2.50,
            },
        )

    def test_reconstructed_template_validates_cleanly(self) -> None:
        from auto_grader.template_schema import validate_template

        family = self._reconstruct()
        self.assertEqual(validate_template(family["template"]), [])

    def test_bad_variant_filename_is_rejected(self) -> None:
        from auto_grader.quiz5_short_answer_reconstruction import _infer_variant_id

        with self.assertRaises(ValueError):
            _infer_variant_id(Path("quiz-five.pdf"))

    def test_capture_raises_when_expected_text_is_missing(self) -> None:
        from auto_grader.quiz5_short_answer_reconstruction import _capture

        with self.assertRaises(ValueError):
            _capture(r"Question (?P<value>.+)", "no matching quiz text here")

    def test_missing_pdf_raises_file_not_found(self) -> None:
        from auto_grader.quiz5_short_answer_reconstruction import _extract_pdf_text

        with self.assertRaises(FileNotFoundError):
            _extract_pdf_text(Path("/tmp/does-not-exist-quiz5.pdf"))
