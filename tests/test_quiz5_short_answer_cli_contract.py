from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import qrcode
import numpy as np


def _asset_root() -> Path:
    configured = os.environ.get("AUTO_GRADER_ASSETS_DIR")
    if configured:
        return Path(configured) / "exams"
    return Path.home() / "dev" / "auto-grader-assets" / "exams"


_ASSET_ROOT = _asset_root()
_QUIZ_A = _ASSET_ROOT / "260326_Quiz _5 A.pdf"
_QUIZ_B = _ASSET_ROOT / "260326_Quiz _5 B.pdf"


@unittest.skipUnless(_QUIZ_A.exists() and _QUIZ_B.exists(), "Quiz #5 legacy PDFs are required for this contract")
class Quiz5ShortAnswerCliContractTests(unittest.TestCase):
    def test_staging_cli_writes_pdfs_and_artifact_json_for_a_b_and_c(self) -> None:
        script_path = Path("scripts/stage_quiz5_short_answer_variants.py")
        self.assertTrue(
            script_path.exists(),
            "Add `scripts/stage_quiz5_short_answer_variants.py` so Quiz #5 can be rendered and staged as real artifacts.",
        )

        with tempfile.TemporaryDirectory(prefix="quiz5-stage-cli-") as tempdir:
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
            self.assertEqual(set(result["variant_ids"]), {"A", "B", "C"})

            for variant_id in ("A", "B", "C"):
                artifact_path = Path(result["artifacts"][variant_id]["artifact_json_path"])
                pdf_path = Path(result["artifacts"][variant_id]["pdf_path"])
                self.assertTrue(artifact_path.exists())
                self.assertTrue(pdf_path.exists())
                self.assertEqual(pdf_path.suffix.lower(), ".pdf")

    def test_ingest_cli_writes_matched_manifest_for_synthetic_scans(self) -> None:
        stage_script_path = Path("scripts/stage_quiz5_short_answer_variants.py")
        ingest_script_path = Path("scripts/ingest_quiz5_short_answer_scans.py")
        self.assertTrue(
            ingest_script_path.exists(),
            "Add `scripts/ingest_quiz5_short_answer_scans.py` so Quiz #5 hand-filled scans can be packaged back into a durable ingest manifest.",
        )

        with tempfile.TemporaryDirectory(prefix="quiz5-stage-ingest-cli-") as tempdir:
            stage_dir = Path(tempdir) / "stage"
            stage_dir.mkdir()
            proc = subprocess.run(
                [
                    str(Path(".venv/bin/python")),
                    str(stage_script_path),
                    "--output-dir",
                    str(stage_dir),
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
            stage_result = json.loads(proc.stdout)

            artifact = json.loads(
                Path(stage_result["artifacts"]["C"]["artifact_json_path"]).read_text(encoding="utf-8")
            )

            scan_dir = Path(tempdir) / "scans"
            scan_dir.mkdir()
            for index, page in enumerate(artifact["pages"], start=1):
                rendered = _render_synthetic_page(page)
                import cv2

                cv2.imwrite(str(scan_dir / f"quiz5-c-p{index}.png"), rendered)

            ingest_dir = Path(tempdir) / "ingest"
            proc = subprocess.run(
                [
                    str(Path(".venv/bin/python")),
                    str(ingest_script_path),
                    "--artifact-json",
                    stage_result["artifacts"]["C"]["artifact_json_path"],
                    "--scan-dir",
                    str(scan_dir),
                    "--output-dir",
                    str(ingest_dir),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            ingest_result = json.loads(proc.stdout)
            manifest = json.loads(Path(ingest_result["manifest_path"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["summary"]["matched"], 2)
            self.assertEqual(manifest["summary"]["unmatched"], 0)


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


if __name__ == "__main__":
    unittest.main()
