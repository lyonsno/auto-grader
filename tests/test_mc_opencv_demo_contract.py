from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import cv2

from auto_grader.paper_calibration_packet import build_mc_paper_calibration_packet
from tests.test_mc_page_extraction_contract import _perspective_distort, _render_marked_page


def _load_demo_module(test_case: unittest.TestCase):
    try:
        from auto_grader.mc_opencv_demo import run_mc_opencv_demo
    except ModuleNotFoundError:
        test_case.fail(
            "Add `auto_grader.mc_opencv_demo.run_mc_opencv_demo(...)` so the landed "
            "MC/OpenCV surfaces can be exercised through one demoable end-to-end runner."
        )
    return run_mc_opencv_demo


class McOpenCvDemoContractTests(unittest.TestCase):
    def test_run_mc_opencv_demo_writes_demoable_batch_result_bundle(self) -> None:
        run_mc_opencv_demo = _load_demo_module(self)
        packet = build_mc_paper_calibration_packet()
        artifact = packet["artifact"]
        page = artifact["pages"][0]

        matched_scan = _perspective_distort(
            _render_marked_page(page, marked_labels={"cal-01": ["B"]})
        )

        with tempfile.TemporaryDirectory(prefix="mc-opencv-demo-contract-") as tempdir:
            temp_path = Path(tempdir)
            artifact_path = temp_path / "artifact.json"
            scans_dir = temp_path / "scans"
            output_dir = temp_path / "output"
            scans_dir.mkdir()

            artifact_path.write_text(json.dumps(artifact), encoding="utf-8")
            cv2.imwrite(str(scans_dir / "matched-page.png"), matched_scan)
            cv2.imwrite(
                str(scans_dir / "unmatched-page.png"),
                255 * matched_scan.astype("uint8") // 255,
            )

            result = run_mc_opencv_demo(
                artifact_json_path=artifact_path,
                scan_dir=scans_dir,
                output_dir=output_dir,
            )

            self.assertEqual(result["artifact_json_path"], str(artifact_path))
            self.assertEqual(result["scan_dir"], str(scans_dir))
            self.assertEqual(result["summary"]["matched"], 1)
            self.assertEqual(result["summary"]["unmatched"], 1)
            self.assertEqual(result["summary"]["ambiguous"], 0)
            self.assertEqual(result["summary"]["review_required_pages"], 0)
            self.assertEqual(
                result["summary"]["scored_question_status_counts"]["correct"],
                1,
                "The demo runner should surface a human-readable score summary instead of "
                "only dumping raw ingest JSON.",
            )

            summary_path = output_dir / "summary.json"
            ingest_path = output_dir / "ingest_result.json"
            matched_dir = output_dir / "matched_pages"
            self.assertTrue(summary_path.exists())
            self.assertTrue(ingest_path.exists())
            self.assertTrue((matched_dir / "matched-page__normalized.png").exists())

            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary, result["summary"])

            ingest_result = json.loads(ingest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                [scan["status"] for scan in ingest_result["scan_results"]],
                ["matched", "unmatched"],
                "The persisted demo bundle should preserve the real ingest outcomes, not a "
                "flattened happy-path summary.",
            )
            self.assertEqual(
                ingest_result["matched_pages"][0]["scored_questions"]["cal-01"]["status"],
                "correct",
            )
            self.assertNotIn(
                "normalized_image",
                ingest_result["scan_results"][1],
                "Unmatched scans should remain tracked failures without pretending they "
                "produced a normalized page artifact.",
            )


if __name__ == "__main__":
    unittest.main()
