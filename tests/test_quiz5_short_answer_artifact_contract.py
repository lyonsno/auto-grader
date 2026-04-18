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
class Quiz5ShortAnswerArtifactContractTests(unittest.TestCase):
    def _family(self):
        from auto_grader.quiz5_short_answer_reconstruction import (
            reconstruct_short_answer_quiz_family,
        )

        return reconstruct_short_answer_quiz_family([_QUIZ_A, _QUIZ_B])

    def test_variant_packet_builder_exists(self) -> None:
        from auto_grader.quiz5_short_answer_packets import (
            build_quiz5_short_answer_variant_packet,
        )

        self.assertTrue(callable(build_quiz5_short_answer_variant_packet))

    def test_variant_packet_builds_qr_backed_two_page_artifact_for_generated_c(self) -> None:
        from auto_grader.quiz5_short_answer_packets import (
            build_quiz5_short_answer_variant_packet,
        )

        artifact = build_quiz5_short_answer_variant_packet(
            self._family(),
            variant_id="C",
            opaque_instance_code="QUIZ5-C-DEMO",
        )

        self.assertEqual(artifact["variant_id"], "C")
        self.assertEqual(artifact["opaque_instance_code"], "QUIZ5-C-DEMO")
        self.assertEqual(len(artifact["pages"]), 2)
        self.assertEqual(
            [page["fallback_page_code"] for page in artifact["pages"]],
            ["QUIZ5-C-DEMO-p1", "QUIZ5-C-DEMO-p2"],
        )
        self.assertEqual(
            [page["page_number"] for page in artifact["pages"]],
            [1, 2],
        )

        page_1 = artifact["pages"][0]
        self.assertEqual(page_1["layout_version"], "quiz5_short_answer_v1")
        self.assertEqual(len(page_1["registration_markers"]), 4)
        self.assertEqual(len(page_1["identity_qr_codes"]), 2)
        self.assertIn(
            "hydrochloric acid",
            " ".join(block["text"] for block in page_1["prompt_blocks"]),
        )

        q1a_box = next(box for box in page_1["response_boxes"] if box["question_id"] == "q1a")
        self.assertEqual(q1a_box["label"], "1a.")

        page_2 = artifact["pages"][1]
        self.assertEqual(
            [box["label"] for box in page_2["response_boxes"]],
            ["5.", "6.[CH4]=", "6. [CCl4]=", "6. [CH2Cl2]="],
        )

    def test_pdf_renderer_emits_pdf_bytes_for_short_answer_packet(self) -> None:
        from auto_grader.pdf_rendering import render_quiz5_short_answer_pdf
        from auto_grader.quiz5_short_answer_packets import (
            build_quiz5_short_answer_variant_packet,
        )

        artifact = build_quiz5_short_answer_variant_packet(
            self._family(),
            variant_id="A",
            opaque_instance_code="QUIZ5-A-DEMO",
        )
        pdf_bytes = render_quiz5_short_answer_pdf(artifact)

        self.assertTrue(pdf_bytes.startswith(b"%PDF-"))
        self.assertIn(b"/Type /Page", pdf_bytes)


if __name__ == "__main__":
    unittest.main()
