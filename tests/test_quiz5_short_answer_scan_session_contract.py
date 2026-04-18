from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

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
class Quiz5ShortAnswerScanSessionContractTests(unittest.TestCase):
    def _artifact(self):
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

    def test_persist_short_answer_scan_session_writes_manifest_and_normalized_images(self) -> None:
        from auto_grader.quiz5_short_answer_scan_session import (
            persist_quiz5_short_answer_scan_session,
        )

        artifact = self._artifact()
        scan_images = {
            "quiz5-p1.png": _render_synthetic_page(artifact["pages"][0]),
            "quiz5-p2.png": _render_synthetic_page(artifact["pages"][1]),
        }

        with tempfile.TemporaryDirectory(prefix="quiz5-short-answer-scan-") as output_dir:
            result = persist_quiz5_short_answer_scan_session(
                scan_images=scan_images,
                artifact=artifact,
                output_dir=output_dir,
            )

            manifest_path = Path(result["manifest_path"])
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

            self.assertEqual(manifest["opaque_instance_code"], artifact["opaque_instance_code"])
            self.assertEqual(manifest["summary"]["matched"], 2)
            self.assertEqual(manifest["summary"]["unmatched"], 0)
            self.assertEqual(manifest["summary"]["ambiguous"], 0)

            normalized_dir = Path(output_dir) / "normalized_images"
            self.assertTrue((normalized_dir / "quiz5-p1.png").exists())
            self.assertTrue((normalized_dir / "quiz5-p2.png").exists())

    def test_persist_short_answer_scan_session_records_unmatched_pages_without_normalized_images(self) -> None:
        from auto_grader.quiz5_short_answer_scan_session import (
            persist_quiz5_short_answer_scan_session,
        )

        artifact = self._artifact()
        blank = np.full((1600, 1200, 3), 255, dtype=np.uint8)

        with tempfile.TemporaryDirectory(prefix="quiz5-short-answer-unmatched-") as output_dir:
            persist_quiz5_short_answer_scan_session(
                scan_images={"blank.png": blank},
                artifact=artifact,
                output_dir=output_dir,
            )

            manifest = json.loads((Path(output_dir) / "session_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["summary"]["matched"], 0)
            self.assertEqual(manifest["summary"]["unmatched"], 1)
            normalized_dir = Path(output_dir) / "normalized_images"
            self.assertFalse(normalized_dir.exists())


if __name__ == "__main__":
    unittest.main()
