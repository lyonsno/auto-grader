"""Contract tests for MC scan-session persistence.

The session persistence layer wraps the in-memory ``ingest_mc_scans`` surface
and writes one durable artifact set per ingest invocation:

- ``session_manifest.json``: machine-readable manifest with checksums,
  per-scan outcomes, and session-level metadata.
- ``normalized_images/<scan_id>.png``: normalized page images for matched scans.
- Re-running the same scan set is idempotent at the artifact-identity level.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from tests.test_mc_page_extraction_contract import (
    _build_artifact,
    _perspective_distort,
    _render_marked_page,
)


def _load_persist_module(test_case: unittest.TestCase):
    try:
        from auto_grader.mc_scan_session import persist_scan_session
    except ModuleNotFoundError:
        test_case.fail(
            "Add `auto_grader.mc_scan_session.persist_scan_session(...)` so the MC "
            "ingest pipeline can write one durable scan-session artifact set to disk."
        )
    return persist_scan_session


class McScanSessionPersistContractTests(unittest.TestCase):
    def _build_matched_scan(self):
        artifact = _build_artifact()
        page = artifact["pages"][0]
        correct_bubble = artifact["answer_key"]["mc-1"]["correct_bubble_label"]
        matched_scan = _perspective_distort(
            _render_marked_page(page, marked_labels={"mc-1": [correct_bubble]})
        )
        return artifact, {"matched-page.png": matched_scan}

    def test_persist_scan_session_writes_manifest_and_normalized_images(self) -> None:
        persist_scan_session = _load_persist_module(self)
        artifact, scan_images = self._build_matched_scan()

        with tempfile.TemporaryDirectory() as output_dir:
            result = persist_scan_session(
                scan_images=scan_images,
                artifact=artifact,
                output_dir=output_dir,
            )

            # Manifest must exist and be valid JSON.
            manifest_path = os.path.join(output_dir, "session_manifest.json")
            self.assertTrue(
                os.path.isfile(manifest_path),
                "persist_scan_session must write a session_manifest.json file.",
            )
            with open(manifest_path) as f:
                manifest = json.load(f)

            # Manifest must carry the instance code and scan results.
            self.assertEqual(
                manifest["opaque_instance_code"],
                artifact["opaque_instance_code"],
            )
            self.assertIn("scan_results", manifest)
            self.assertIsInstance(manifest["scan_results"], list)
            self.assertEqual(len(manifest["scan_results"]), 1)

            scan_entry = manifest["scan_results"][0]
            self.assertEqual(scan_entry["scan_id"], "matched-page.png")
            self.assertEqual(scan_entry["status"], "matched")
            self.assertIsNone(scan_entry["failure_reason"])
            self.assertIn("checksum", scan_entry)
            self.assertRegex(scan_entry["checksum"], r"^[0-9a-f]{64}$")

            # Matched scan must have a normalized image written to disk.
            normalized_dir = os.path.join(output_dir, "normalized_images")
            self.assertTrue(
                os.path.isdir(normalized_dir),
                "persist_scan_session must create a normalized_images/ directory.",
            )
            expected_image_path = os.path.join(normalized_dir, "matched-page.png")
            self.assertTrue(
                os.path.isfile(expected_image_path),
                "Each matched scan must have its normalized image written to "
                "normalized_images/<scan_id>.png.",
            )

            # Manifest scan entry must carry scored_questions for matched scans.
            self.assertIn("scored_questions", scan_entry)
            self.assertEqual(
                scan_entry["scored_questions"]["mc-1"]["status"], "correct"
            )

            # Return value must include the manifest path and output dir.
            self.assertEqual(result["output_dir"], output_dir)
            self.assertEqual(result["manifest_path"], manifest_path)

    def test_persist_scan_session_records_unmatched_scans_without_images(self) -> None:
        persist_scan_session = _load_persist_module(self)
        artifact = _build_artifact()
        blank_scan = np.full((600, 900, 3), 255, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as output_dir:
            persist_scan_session(
                scan_images={"blank-page.png": blank_scan},
                artifact=artifact,
                output_dir=output_dir,
            )

            with open(os.path.join(output_dir, "session_manifest.json")) as f:
                manifest = json.load(f)

            self.assertEqual(len(manifest["scan_results"]), 1)
            scan_entry = manifest["scan_results"][0]
            self.assertEqual(scan_entry["scan_id"], "blank-page.png")
            self.assertEqual(scan_entry["status"], "unmatched")
            self.assertIsNotNone(scan_entry["failure_reason"])
            self.assertNotIn("scored_questions", scan_entry)

            # No normalized image for unmatched scans.
            normalized_dir = os.path.join(output_dir, "normalized_images")
            if os.path.isdir(normalized_dir):
                self.assertEqual(
                    os.listdir(normalized_dir),
                    [],
                    "Unmatched scans must not produce normalized image files.",
                )

    def test_persist_scan_session_is_idempotent(self) -> None:
        persist_scan_session = _load_persist_module(self)
        artifact, scan_images = self._build_matched_scan()

        with tempfile.TemporaryDirectory() as output_dir:
            result_1 = persist_scan_session(
                scan_images=scan_images,
                artifact=artifact,
                output_dir=output_dir,
            )

            with open(result_1["manifest_path"]) as f:
                manifest_1 = json.load(f)

            # Run again with the same inputs into the same directory.
            result_2 = persist_scan_session(
                scan_images=scan_images,
                artifact=artifact,
                output_dir=output_dir,
            )

            with open(result_2["manifest_path"]) as f:
                manifest_2 = json.load(f)

            # Manifests must be identical — no duplicated entries,
            # no corrupted state from the second run.
            self.assertEqual(
                manifest_1["scan_results"],
                manifest_2["scan_results"],
                "Re-running persist_scan_session with the same inputs must produce "
                "an identical manifest, not duplicate or corrupt the session record.",
            )

    def test_persist_scan_session_manifest_carries_expected_page_codes(self) -> None:
        persist_scan_session = _load_persist_module(self)
        artifact, scan_images = self._build_matched_scan()

        with tempfile.TemporaryDirectory() as output_dir:
            persist_scan_session(
                scan_images=scan_images,
                artifact=artifact,
                output_dir=output_dir,
            )

            with open(os.path.join(output_dir, "session_manifest.json")) as f:
                manifest = json.load(f)

            self.assertIn("expected_page_codes", manifest)
            self.assertIsInstance(manifest["expected_page_codes"], list)
            self.assertEqual(
                manifest["expected_page_codes"],
                [artifact["pages"][0]["fallback_page_code"]],
            )

    def test_persist_scan_session_manifest_carries_summary_counts(self) -> None:
        persist_scan_session = _load_persist_module(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]
        correct_bubble = artifact["answer_key"]["mc-1"]["correct_bubble_label"]
        matched_scan = _perspective_distort(
            _render_marked_page(page, marked_labels={"mc-1": [correct_bubble]})
        )
        blank_scan = np.full((600, 900, 3), 255, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as output_dir:
            persist_scan_session(
                scan_images={
                    "matched.png": matched_scan,
                    "blank.png": blank_scan,
                },
                artifact=artifact,
                output_dir=output_dir,
            )

            with open(os.path.join(output_dir, "session_manifest.json")) as f:
                manifest = json.load(f)

            self.assertIn("summary", manifest)
            summary = manifest["summary"]
            self.assertEqual(summary["total_scans"], 2)
            self.assertEqual(summary["matched"], 1)
            self.assertEqual(summary["unmatched"], 1)
            self.assertEqual(summary["ambiguous"], 0)
            self.assertEqual(summary["review_required"], 0)


if __name__ == "__main__":
    unittest.main()
