from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import qrcode


def _asset_root() -> Path:
    configured = os.environ.get("AUTO_GRADER_ASSETS_DIR")
    if configured:
        return Path(configured) / "exams"
    return Path.home() / "dev" / "auto-grader-assets" / "exams"


_ASSET_ROOT = _asset_root()
_QUIZ_A = _ASSET_ROOT / "260326_Quiz _5 A.pdf"
_QUIZ_B = _ASSET_ROOT / "260326_Quiz _5 B.pdf"


def _render_synthetic_page(artifact_page: dict, *, scale: float = 4.0) -> np.ndarray:
    from PIL import Image, ImageDraw

    width = int(artifact_page["width"] * scale)
    height = int(artifact_page["height"] * scale)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for marker in artifact_page["registration_markers"]:
        draw.rectangle(
            [
                marker["x"] * scale,
                marker["y"] * scale,
                (marker["x"] + marker["width"]) * scale,
                (marker["y"] + marker["height"]) * scale,
            ],
            fill="black",
        )

    correction_levels = {
        "L": qrcode.constants.ERROR_CORRECT_L,
        "M": qrcode.constants.ERROR_CORRECT_M,
        "Q": qrcode.constants.ERROR_CORRECT_Q,
        "H": qrcode.constants.ERROR_CORRECT_H,
    }
    for qr_code in artifact_page["identity_qr_codes"]:
        qr = qrcode.QRCode(
            border=qr_code["border_modules"],
            error_correction=correction_levels[qr_code["error_correction"]],
            box_size=8,
        )
        qr.add_data(qr_code["payload"])
        qr.make(fit=True)
        qr_image = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        qr_image = qr_image.resize(
            (int(qr_code["width"] * scale), int(qr_code["height"] * scale))
        )
        canvas.paste(qr_image, (int(qr_code["x"] * scale), int(qr_code["y"] * scale)))

    return np.array(canvas)


@unittest.skipUnless(_QUIZ_A.exists() and _QUIZ_B.exists(), "Quiz #5 legacy PDFs are required for this contract")
class Quiz5ShortAnswerTrialPrepContractTests(unittest.TestCase):
    def _artifact(self) -> dict:
        from auto_grader.quiz5_short_answer_packets import (
            build_quiz5_short_answer_variant_packet,
        )
        from auto_grader.quiz5_short_answer_reconstruction import (
            reconstruct_short_answer_quiz_family,
        )

        family = reconstruct_short_answer_quiz_family([_QUIZ_A, _QUIZ_B])
        return build_quiz5_short_answer_variant_packet(
            family,
            variant_id="C",
            opaque_instance_code="QUIZ5-C-DEMO",
        )

    def test_trial_prep_writes_per_response_box_crops_and_metadata(self) -> None:
        from auto_grader.quiz5_short_answer_trial_prep import (
            prepare_quiz5_short_answer_trial_crops,
        )

        artifact = self._artifact()
        manifest = {
            "opaque_instance_code": artifact["opaque_instance_code"],
            "expected_page_codes": [page["fallback_page_code"] for page in artifact["pages"]],
            "scan_results": [
                {
                    "scan_id": "quiz5-p1.png",
                    "checksum": "checksum-p1",
                    "status": "matched",
                    "failure_reason": None,
                    "page_number": 1,
                    "fallback_page_code": artifact["pages"][0]["fallback_page_code"],
                },
                {
                    "scan_id": "quiz5-p2.png",
                    "checksum": "checksum-p2",
                    "status": "matched",
                    "failure_reason": None,
                    "page_number": 2,
                    "fallback_page_code": artifact["pages"][1]["fallback_page_code"],
                },
            ],
            "summary": {"total_scans": 2, "matched": 2, "unmatched": 0, "ambiguous": 0},
        }

        with tempfile.TemporaryDirectory(prefix="quiz5-trial-prep-") as root:
            root_path = Path(root)
            normalized_dir = root_path / "normalized_images"
            normalized_dir.mkdir()
            cv2.imwrite(
                str(normalized_dir / "quiz5-p1.png"),
                _render_synthetic_page(artifact["pages"][0]),
            )
            cv2.imwrite(
                str(normalized_dir / "quiz5-p2.png"),
                _render_synthetic_page(artifact["pages"][1]),
            )

            result = prepare_quiz5_short_answer_trial_crops(
                artifact=artifact,
                manifest=manifest,
                normalized_dir=normalized_dir,
                output_dir=root_path / "trial_prep",
            )

            manifest_path = Path(result["manifest_path"])
            self.assertTrue(manifest_path.exists())
            written = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(written["opaque_instance_code"], "QUIZ5-C-DEMO")
            self.assertEqual(written["summary"]["matched_pages"], 2)
            self.assertEqual(written["summary"]["response_box_crops"], 9)

            entries = written["responses"]
            self.assertEqual(len(entries), 9)
            self.assertEqual(entries[0]["question_id"], "q1a")
            self.assertEqual(entries[-1]["question_id"], "q6-ch2cl2")
            self.assertTrue(Path(entries[0]["crop_path"]).exists())

            q6_entries = [entry for entry in entries if entry["question_id"].startswith("q6-")]
            self.assertEqual([entry["question_id"] for entry in q6_entries], ["q6-ch4", "q6-ccl4", "q6-ch2cl2"])
            self.assertEqual(len({entry["crop_box"]["y"] for entry in q6_entries}), 3)
            self.assertTrue(all(entry["crop_box"]["height"] > 0 for entry in q6_entries))

    def test_trial_prep_rejects_missing_normalized_image_for_matched_page(self) -> None:
        from auto_grader.quiz5_short_answer_trial_prep import (
            prepare_quiz5_short_answer_trial_crops,
        )

        artifact = self._artifact()
        manifest = {
            "opaque_instance_code": artifact["opaque_instance_code"],
            "expected_page_codes": [artifact["pages"][0]["fallback_page_code"]],
            "scan_results": [
                {
                    "scan_id": "missing.png",
                    "checksum": "checksum-p1",
                    "status": "matched",
                    "failure_reason": None,
                    "page_number": 1,
                    "fallback_page_code": artifact["pages"][0]["fallback_page_code"],
                }
            ],
            "summary": {"total_scans": 1, "matched": 1, "unmatched": 0, "ambiguous": 0},
        }

        with tempfile.TemporaryDirectory(prefix="quiz5-trial-prep-missing-") as root:
            with self.assertRaisesRegex(ValueError, "Missing normalized image"):
                prepare_quiz5_short_answer_trial_crops(
                    artifact=artifact,
                    manifest=manifest,
                    normalized_dir=Path(root) / "normalized_images",
                    output_dir=Path(root) / "trial_prep",
                )


if __name__ == "__main__":
    unittest.main()
