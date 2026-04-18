"""DB-backed contracts for returned Quiz #5 short-answer scan-session persistence."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
import uuid

import numpy as np
import qrcode

try:
    import psycopg
    from psycopg import sql
    from psycopg.rows import dict_row
except ModuleNotFoundError:
    psycopg = None
    sql = None
    dict_row = None

from auto_grader import db as db_module
from auto_grader.db import initialize_schema


def _asset_root() -> Path:
    configured = os.environ.get("AUTO_GRADER_ASSETS_DIR")
    if configured:
        return Path(configured) / "exams"
    return Path.home() / "dev" / "auto-grader-assets" / "exams"


def _postgres_test_database_url() -> str | None:
    return os.environ.get("TEST_DATABASE_URL")


_ASSET_ROOT = _asset_root()
_QUIZ_A = _ASSET_ROOT / "260326_Quiz _5 A.pdf"
_QUIZ_B = _ASSET_ROOT / "260326_Quiz _5 B.pdf"


@unittest.skipUnless(_QUIZ_A.exists() and _QUIZ_B.exists(), "Quiz #5 legacy PDFs are required for this contract")
class Quiz5ShortAnswerScanDbContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.database_url = _postgres_test_database_url()
        if cls.database_url is None:
            raise unittest.SkipTest("Set TEST_DATABASE_URL to run Quiz #5 short-answer scan DB contract tests.")
        try:
            cls.database_url = db_module._normalize_postgres_database_url(
                cls.database_url, label="TEST_DATABASE_URL"
            )
        except ValueError as exc:
            raise AssertionError(str(exc)) from exc
        if psycopg is None:
            raise AssertionError("Quiz #5 short-answer scan DB contract tests require psycopg.")

    def setUp(self) -> None:
        self.database_url = self.__class__.database_url
        self.schema_name = f"ag_quiz5_short_answer_scan_{uuid.uuid4().hex}"
        self.connection = None
        self._schema_created = False

        try:
            with psycopg.connect(self.database_url, autocommit=True) as admin_conn:
                admin_conn.execute(
                    sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema_name))
                )
            self._schema_created = True
            self.connection = psycopg.connect(
                self.database_url, autocommit=True, row_factory=dict_row
            )
            self.connection.execute(
                sql.SQL("SET search_path TO {}, public").format(
                    sql.Identifier(self.schema_name)
                )
            )
        except Exception as exc:
            self._cleanup_schema()
            raise AssertionError(f"Setup failed: {exc}") from exc

        initialize_schema(self.connection)
        self._register_variant_packet()

    def tearDown(self) -> None:
        if self.connection is not None:
            self.connection.close()
        self._cleanup_schema()

    def _cleanup_schema(self) -> None:
        if not self._schema_created:
            return
        try:
            with psycopg.connect(self.database_url, autocommit=True) as admin_conn:
                admin_conn.execute(
                    sql.SQL("DROP SCHEMA {} CASCADE").format(
                        sql.Identifier(self.schema_name)
                    )
                )
        except Exception:
            pass

    def _register_variant_packet(self) -> None:
        from auto_grader.quiz5_short_answer_packets import (
            build_quiz5_short_answer_variant_packet,
            register_quiz5_short_answer_variants,
        )
        from auto_grader.quiz5_short_answer_reconstruction import (
            reconstruct_short_answer_quiz_family,
        )

        family = reconstruct_short_answer_quiz_family([_QUIZ_A, _QUIZ_B])
        self.packet = build_quiz5_short_answer_variant_packet(
            family, variant_id="C", opaque_instance_code="QUIZ5-C"
        )
        register_quiz5_short_answer_variants(
            family=family,
            packets={"C": self.packet},
            connection=self.connection,
        )
        self.exam_instance_id = self.connection.execute(
            "SELECT id FROM exam_instances WHERE opaque_instance_code = 'QUIZ5-C'"
        ).fetchone()["id"]

    def _manifest(self) -> dict:
        from auto_grader.quiz5_short_answer_scan_session import (
            persist_quiz5_short_answer_scan_session,
        )

        with tempfile.TemporaryDirectory(prefix="quiz5-short-answer-scan-db-") as output_dir:
            scan_images = {
                "quiz5-c-p1.png": _render_synthetic_page(self.packet["pages"][0]),
                "quiz5-c-p2.png": _render_synthetic_page(self.packet["pages"][1]),
            }
            result = persist_quiz5_short_answer_scan_session(
                scan_images=scan_images,
                artifact=self.packet,
                output_dir=output_dir,
            )
            return json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))

    def test_persist_scan_session_manifest_to_db_records_scan_artifacts_and_audit_event(self) -> None:
        from auto_grader.quiz5_short_answer_scan_db import (
            persist_quiz5_short_answer_scan_session_manifest_to_db,
        )

        manifest = self._manifest()
        result = persist_quiz5_short_answer_scan_session_manifest_to_db(
            manifest=manifest,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        self.assertTrue(result["created"])
        self.assertEqual(result["exam_instance_id"], self.exam_instance_id)
        self.assertEqual(result["summary"]["matched"], 2)
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) AS n FROM scan_artifacts").fetchone()["n"],
            2,
        )
        self.assertEqual(
            self.connection.execute(
                """
                SELECT COUNT(*) AS n
                FROM audit_events
                WHERE entity_type = 'exam_instance'
                  AND entity_id = %s
                  AND event_type = 'quiz5_short_answer_scan_session_persisted'
                """,
                (self.exam_instance_id,),
            ).fetchone()["n"],
            1,
        )

    def test_persist_scan_session_manifest_to_db_is_idempotent_for_same_manifest(self) -> None:
        from auto_grader.quiz5_short_answer_scan_db import (
            persist_quiz5_short_answer_scan_session_manifest_to_db,
        )

        manifest = self._manifest()
        first = persist_quiz5_short_answer_scan_session_manifest_to_db(
            manifest=manifest,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )
        second = persist_quiz5_short_answer_scan_session_manifest_to_db(
            manifest=manifest,
            exam_instance_id=self.exam_instance_id,
            connection=self.connection,
        )

        self.assertTrue(first["created"])
        self.assertFalse(second["created"])
        self.assertEqual(
            self.connection.execute("SELECT COUNT(*) AS n FROM scan_artifacts").fetchone()["n"],
            2,
        )
        self.assertEqual(
            self.connection.execute(
                "SELECT COUNT(*) AS n FROM audit_events WHERE event_type = 'quiz5_short_answer_scan_session_persisted'"
            ).fetchone()["n"],
            1,
        )


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
