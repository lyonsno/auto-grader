from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import cv2

from tests.test_mc_page_extraction_contract import _perspective_distort, _render_marked_page


_TEMPLATE_PATH = Path("templates/chm141-final-fall2023.yaml")


def _load_builders(test_case: unittest.TestCase):
    try:
        from auto_grader.generated_exam_demo import (
            build_generated_mc_exam_demo_packet,
            write_generated_mc_exam_demo_packet,
        )
    except ModuleNotFoundError:
        test_case.fail(
            "Add `auto_grader.generated_exam_demo` so we can generate a real "
            "printable MC exam packet from an actual chemistry template instead "
            "of only calibrating the OpenCV lane against synthetic packets."
        )
    except ImportError:
        test_case.fail(
            "Export `build_generated_mc_exam_demo_packet(...)` and "
            "`write_generated_mc_exam_demo_packet(...)` so the generated-exam "
            "demo packet has a stable repo-local surface."
        )
    return build_generated_mc_exam_demo_packet, write_generated_mc_exam_demo_packet


def _load_demo_runner(test_case: unittest.TestCase):
    try:
        from auto_grader.mc_opencv_demo import run_mc_opencv_demo
    except ModuleNotFoundError:
        test_case.fail(
            "Keep `auto_grader.mc_opencv_demo.run_mc_opencv_demo(...)` available so "
            "the generated-exam demo packet can be exercised through the landed "
            "end-to-end MC/OpenCV runner."
        )
    return run_mc_opencv_demo


class GeneratedExamDemoContractTests(unittest.TestCase):
    def test_builder_uses_real_chemistry_template_for_printable_mc_demo_packet(self) -> None:
        build_generated_mc_exam_demo_packet, _ = _load_builders(self)

        packet = build_generated_mc_exam_demo_packet(
            template_path=_TEMPLATE_PATH,
            student_id="demo-001",
            student_name="Demo Student",
            attempt_number=1,
            seed=17,
        )

        artifact = packet["artifact"]
        self.assertEqual(
            Path(packet["template_path"]).name,
            _TEMPLATE_PATH.name,
            "The packet should remember which real template file it came from "
            "without depending on the caller's working directory.",
        )
        self.assertEqual(artifact["template_slug"], "chm141-final-fall2023")
        self.assertEqual(len(artifact["mc_questions"]), 19)
        self.assertEqual(len(artifact["pages"]), 5)
        self.assertEqual(artifact["student_id"], "demo-001")
        self.assertTrue(
            artifact["mc_questions"][0]["prompt"].startswith(
                "Which one of the following substances is classified as an element?"
            ),
            "The generated-exam demo packet should preserve the real chemistry "
            "question text instead of replacing it with calibration instructions.",
        )

    def test_writer_emits_pdf_artifact_and_metadata_for_generated_exam_demo(self) -> None:
        _, write_generated_mc_exam_demo_packet = _load_builders(self)

        with tempfile.TemporaryDirectory(prefix="generated-exam-demo-contract-") as tempdir:
            result = write_generated_mc_exam_demo_packet(
                output_dir=tempdir,
                template_path=_TEMPLATE_PATH,
                student_id="demo-001",
                student_name="Demo Student",
                attempt_number=1,
                seed=17,
            )

            pdf_path = Path(result["pdf_path"])
            artifact_path = Path(result["artifact_path"])
            metadata_path = Path(result["metadata_path"])

            self.assertTrue(pdf_path.exists())
            self.assertTrue(artifact_path.exists())
            self.assertTrue(metadata_path.exists())
            self.assertTrue(pdf_path.read_bytes().startswith(b"%PDF"))
            self.assertEqual(result["page_count"], 5)
            self.assertEqual(result["question_count"], 19)

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["template_slug"], "chm141-final-fall2023")
            self.assertEqual(metadata["student_id"], "demo-001")
            self.assertEqual(metadata["attempt_number"], 1)
            self.assertEqual(
                Path(metadata["artifact_path"]).name,
                artifact_path.name,
                "The metadata should point at the generated artifact JSON so the "
                "later demo-runner step can reuse the same packet cleanly.",
            )

    def test_generated_exam_packet_runs_through_landed_demo_runner(self) -> None:
        build_generated_mc_exam_demo_packet, _ = _load_builders(self)
        run_mc_opencv_demo = _load_demo_runner(self)

        packet = build_generated_mc_exam_demo_packet(
            template_path=_TEMPLATE_PATH,
            student_id="demo-001",
            student_name="Demo Student",
            attempt_number=1,
            seed=17,
        )
        artifact = packet["artifact"]
        first_page = artifact["pages"][0]
        first_page_question_ids = {
            region["question_id"] for region in first_page["bubble_regions"]
        }
        first_question_id = artifact["mc_questions"][0]["question_id"]
        self.assertIn(first_question_id, first_page_question_ids)
        correct_bubble_label = artifact["answer_key"][first_question_id]["correct_bubble_label"]

        marked_scan = _perspective_distort(
            _render_marked_page(first_page, marked_labels={first_question_id: [correct_bubble_label]})
        )

        with tempfile.TemporaryDirectory(prefix="generated-exam-demo-runner-") as tempdir:
            temp_path = Path(tempdir)
            artifact_path = temp_path / "artifact.json"
            scans_dir = temp_path / "scans"
            output_dir = temp_path / "output"
            scans_dir.mkdir()

            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            cv2.imwrite(str(scans_dir / "generated-page.png"), marked_scan)

            result = run_mc_opencv_demo(
                artifact_json_path=artifact_path,
                scan_dir=scans_dir,
                output_dir=output_dir,
            )

            self.assertEqual(result["summary"]["matched"], 1)
            self.assertEqual(result["summary"]["unmatched"], 0)
            self.assertEqual(
                result["summary"]["scored_question_status_counts"]["correct"],
                1,
                "A real generated exam page should flow through the landed demo "
                "runner coherently, not just the special calibration packets.",
            )
            self.assertEqual(
                result["summary"]["scored_question_status_counts"]["blank"],
                len(first_page_question_ids) - 1,
            )


if __name__ == "__main__":
    unittest.main()
