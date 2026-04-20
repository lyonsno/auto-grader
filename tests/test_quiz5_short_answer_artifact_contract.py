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
        self.assertEqual(page_1["reference_template_variant_id"], "A")
        self.assertEqual(page_1["template_page_number"], 1)
        self.assertEqual(len(page_1["registration_markers"]), 4)
        self.assertEqual(len(page_1["identity_qr_codes"]), 2)
        self.assertIn(
            "hydrochloric acid",
            " ".join(block["text"] for block in page_1["prompt_blocks"]),
        )

        q1a_box = next(box for box in page_1["response_boxes"] if box["question_id"] == "q1a")
        self.assertEqual(q1a_box["label"], "1a.")

        page_2 = artifact["pages"][1]
        self.assertEqual(page_2["reference_template_variant_id"], "A")
        self.assertEqual(page_2["template_page_number"], 2)
        self.assertEqual(
            [box["label"] for box in page_2["response_boxes"]],
            ["5.", "6.[CH4]=", "6. [CCl4]=", "6. [CH2Cl2]="],
        )

    def test_identity_qr_codes_keep_the_larger_smoke_safe_size(self) -> None:
        from auto_grader.quiz5_short_answer_packets import (
            build_quiz5_short_answer_variant_packet,
        )

        artifact = build_quiz5_short_answer_variant_packet(
            self._family(),
            variant_id="C",
            opaque_instance_code="QUIZ5-C-DEMO",
        )

        qr_codes = artifact["pages"][0]["identity_qr_codes"]
        self.assertEqual([qr["width"] for qr in qr_codes], [30, 30])
        self.assertEqual([qr["height"] for qr in qr_codes], [30, 30])
        self.assertTrue(all(qr["y"] >= 44 for qr in qr_codes))

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
        self.assertTrue(b"/Type/Page" in pdf_bytes or b"/Type /Page" in pdf_bytes)

    def test_pdf_renderer_uses_source_quiz_identity_instead_of_internal_packet_header(self) -> None:
        import fitz

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
        with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
            extracted_text = "\n".join(page.get_text("text") for page in document)

        self.assertIn("CHM 142", extracted_text)
        self.assertIn("Prof. Lyons", extracted_text)
        self.assertIn("Quiz #5", extracted_text)
        self.assertIn("Name:", extracted_text)
        self.assertIn("Be sure to enter your answers inside the boxes !", extracted_text)
        self.assertNotIn("Quiz 5 Short Answer", extracted_text)
        self.assertNotIn("Instance: QUIZ5-A-DEMO", extracted_text)
        self.assertNotIn("Page code: QUIZ5-A-DEMO-p1", extracted_text)

    def test_page_one_geometry_tracks_source_quiz_rhythm(self) -> None:
        from auto_grader.quiz5_short_answer_packets import (
            build_quiz5_short_answer_variant_packet,
        )

        artifact = build_quiz5_short_answer_variant_packet(
            self._family(),
            variant_id="A",
            opaque_instance_code="QUIZ5-A-DEMO",
        )

        page_1 = artifact["pages"][0]
        prompts = {block["question_id"]: block for block in page_1["prompt_blocks"]}
        boxes = {box["question_id"]: box for box in page_1["response_boxes"]}

        self.assertEqual(prompts["q1a"]["label_text"], "a.")
        self.assertEqual(prompts["q1b"]["label_text"], "b.")
        self.assertLess(prompts["q1a"]["label_x"], prompts["q1a"]["x"])
        self.assertLess(prompts["q1b"]["label_x"], prompts["q1b"]["x"])
        self.assertLess(prompts["q1a"]["x"], 165)
        self.assertLess(prompts["q1b"]["x"], 165)
        self.assertGreater(prompts["q1a"]["y"], 372)
        self.assertGreater(prompts["q1b"]["y"], 450)
        self.assertLess(prompts["q1a"]["y"], boxes["q1a"]["y"])
        self.assertLess(prompts["q1b"]["y"], boxes["q1b"]["y"])
        self.assertGreaterEqual(prompts["q1a"]["continuation_x"], prompts["q1a"]["x"] + 48)
        self.assertGreaterEqual(prompts["q1b"]["continuation_x"], prompts["q1b"]["x"] + 48)
        self.assertEqual(
            prompts["q1a"]["wrapped_lines"],
            [
                "Write a net ionic equation to show how methylamine CH3NH2 behaves as a",
                "Bronsted base in water.",
            ],
        )
        self.assertEqual(
            prompts["q1b"]["wrapped_lines"],
            [
                "Write a net ionic equation to show how acetic acid CH3COOH behaves as a",
                "Bronsted acid in water.",
            ],
        )
        self.assertLess(boxes["q4"]["y"] + boxes["q4"]["height"], 744)

    def test_generated_c_page_one_preserves_source_layout_by_replacing_only_changed_substrings(self) -> None:
        from auto_grader.quiz5_short_answer_packets import (
            build_quiz5_short_answer_variant_packet,
        )

        artifact = build_quiz5_short_answer_variant_packet(
            self._family(),
            variant_id="C",
            opaque_instance_code="QUIZ5-C-DEMO",
        )

        page_1 = artifact["pages"][0]
        overlays = page_1["text_overlays"]
        self.assertEqual(
            [overlay["search_text"] for overlay in overlays],
            [
                "Write a net ionic equation to show how dimethylamine (CH3)2NH behaves as a",
                "Write a net ionic equation to show how isobutyric acid (CH3)2COOH behaves as",
                "What is the pH of an aqueous solution of 1.68×10-2 M hydroiodic acid?",
                "What is the pH of a 0.0589 M aqueous solution of sodium hydroxide?",
                "2.00?",
            ],
        )
        self.assertEqual(
            overlays[0]["replacement_text"],
            "Write a net ionic equation to show how methylamine CH3NH2 behaves as a",
        )
        self.assertIn("CH<sub>3</sub>NH<sub>2</sub>", overlays[0]["replacement_html"])
        self.assertEqual(
            overlays[1]["replacement_text"],
            "Write a net ionic equation to show how acetic acid CH3COOH behaves as",
        )
        self.assertIn("CH<sub>3</sub>COOH", overlays[1]["replacement_html"])
        self.assertEqual(
            overlays[2]["replacement_text"],
            "What is the pH of an aqueous solution of 9.74×10-3 M hydrochloric acid?",
        )
        self.assertIn("9.74×10<sup>-3</sup>", overlays[2]["replacement_html"])
        self.assertEqual(overlays[3]["replacement_text"], "What is the pH of a 0.0339 M aqueous solution of sodium hydroxide?")
        self.assertEqual(overlays[4]["replacement_text"], "2.25?")

    def test_generated_c_page_two_preserves_source_layout_by_replacing_only_kc_values(self) -> None:
        from auto_grader.quiz5_short_answer_packets import (
            build_quiz5_short_answer_variant_packet,
        )

        artifact = build_quiz5_short_answer_variant_packet(
            self._family(),
            variant_id="C",
            opaque_instance_code="QUIZ5-C-DEMO",
        )

        page_2 = artifact["pages"][1]
        overlays = page_2["text_overlays"]
        self.assertEqual(
            [overlay["search_text"] for overlay in overlays],
            ["0.0029", "0.0952"],
        )
        self.assertEqual(
            [overlay["replacement_text"] for overlay in overlays],
            ["0.0039", "0.0714"],
        )

if __name__ == "__main__":
    unittest.main()
