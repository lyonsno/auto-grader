from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from tests.test_mc_page_extraction_contract import (
    _build_artifact,
    _perspective_distort,
    _render_marked_page,
)


def _load_ingest_module(test_case: unittest.TestCase):
    try:
        from auto_grader.mc_scan_ingest import ingest_mc_scans
    except ModuleNotFoundError:
        test_case.fail(
            "Add `auto_grader.mc_scan_ingest.ingest_mc_scans(...)` so the MC/OpenCV "
            "lane has one scan-level packaging surface above matched-page extraction."
        )
    return ingest_mc_scans


class McScanIngestContractTests(unittest.TestCase):
    def test_ingest_mc_scans_packages_matched_and_unmatched_pages_without_guessing(self) -> None:
        ingest_mc_scans = _load_ingest_module(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]
        correct_bubble = artifact["answer_key"]["mc-1"]["correct_bubble_label"]

        matched_scan = _perspective_distort(
            _render_marked_page(page, marked_labels={"mc-1": [correct_bubble]})
        )
        unmatched_scan = np.full_like(matched_scan, 255)

        packaged = ingest_mc_scans(
            {
                "matched-page.png": matched_scan,
                "unmatched-page.png": unmatched_scan,
            },
            artifact,
        )

        self.assertEqual(packaged["opaque_instance_code"], artifact["opaque_instance_code"])
        self.assertEqual(packaged["expected_page_codes"], [page["fallback_page_code"]])
        self.assertEqual(
            [result["status"] for result in packaged["scan_results"]],
            ["matched", "unmatched"],
            "The scan-ingest surface should preserve both successful and failed identity outcomes "
            "instead of only returning the happy matched path.",
        )
        matched_result = packaged["scan_results"][0]
        self.assertEqual(matched_result["scan_id"], "matched-page.png")
        self.assertEqual(matched_result["status"], "matched")
        self.assertIsNone(matched_result["failure_reason"])
        self.assertEqual(matched_result["page_number"], page["page_number"])
        self.assertEqual(matched_result["fallback_page_code"], page["fallback_page_code"])
        self.assertEqual(matched_result["scored_questions"]["mc-1"]["status"], "correct")
        self.assertIn("normalized_image", matched_result)
        self.assertTrue(matched_result["checksum"])

        unmatched_result = packaged["scan_results"][1]
        self.assertEqual(unmatched_result["scan_id"], "unmatched-page.png")
        self.assertEqual(unmatched_result["status"], "unmatched")
        self.assertRegex(
            unmatched_result["failure_reason"],
            r"(?is)no page-identity qr code",
            "Unreadable scans should remain tracked unmatched artifacts with an explicit reason.",
        )
        self.assertNotIn("normalized_image", unmatched_result)
        self.assertEqual([result["scan_id"] for result in packaged["matched_pages"]], ["matched-page.png"])
        self.assertEqual(
            [result["scan_id"] for result in packaged["unmatched_scans"]],
            ["unmatched-page.png"],
        )
        self.assertEqual(packaged["ambiguous_scans"], [])
        self.assertEqual(packaged["review_required_pages"], [])

    def test_ingest_mc_scans_marks_duplicate_page_matches_as_ambiguous(self) -> None:
        ingest_mc_scans = _load_ingest_module(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]
        correct_bubble = artifact["answer_key"]["mc-1"]["correct_bubble_label"]

        scan_a = _perspective_distort(
            _render_marked_page(page, marked_labels={"mc-1": [correct_bubble]})
        )
        scan_b = _perspective_distort(
            _render_marked_page(page, marked_labels={"mc-1": [correct_bubble]})
        )

        packaged = ingest_mc_scans(
            {
                "page-a.png": scan_a,
                "page-b.png": scan_b,
            },
            artifact,
        )

        self.assertEqual(packaged["matched_pages"], [])
        self.assertEqual(
            [result["status"] for result in packaged["scan_results"]],
            ["ambiguous", "ambiguous"],
            "If two scans claim the same page code in one ingest batch, the packaging layer "
            "should not silently pick a winner.",
        )
        for result in packaged["scan_results"]:
            self.assertRegex(
                result["failure_reason"],
                r"(?is)multiple scans matched same page code",
            )
            self.assertEqual(result["fallback_page_code"], page["fallback_page_code"])
            self.assertNotIn("normalized_image", result)
        self.assertEqual(
            [result["scan_id"] for result in packaged["ambiguous_scans"]],
            ["page-a.png", "page-b.png"],
        )

    def test_ingest_mc_scans_collects_review_required_matches(self) -> None:
        ingest_mc_scans = _load_ingest_module(self)
        artifact = _build_artifact()
        page = artifact["pages"][0]

        review_scan = _perspective_distort(
            _render_marked_page(page, marked_labels={"mc-1": ["A", "D"]})
        )

        packaged = ingest_mc_scans({"review-page.png": review_scan}, artifact)

        self.assertEqual([result["status"] for result in packaged["scan_results"]], ["matched"])
        self.assertEqual(
            packaged["scan_results"][0]["scored_questions"]["mc-1"]["status"],
            "multiple_marked",
        )
        self.assertTrue(packaged["scan_results"][0]["scored_questions"]["mc-1"]["review_required"])
        self.assertEqual(
            [result["scan_id"] for result in packaged["review_required_pages"]],
            ["review-page.png"],
            "The scan-level packaging surface should surface review-required matched pages "
            "without forcing downstream callers to re-scan every scored question by hand.",
        )

    def test_ingest_mc_scans_marks_conflicting_qr_payloads_as_ambiguous(self) -> None:
        ingest_mc_scans = _load_ingest_module(self)
        artifact = _build_artifact()

        from tests.test_scan_readback_contract import _synthetic_page

        conflicting_scan = _synthetic_page("inst_test-p1", "inst_test-p2")

        packaged = ingest_mc_scans({"conflicting.png": conflicting_scan}, artifact)

        self.assertEqual([result["status"] for result in packaged["scan_results"]], ["ambiguous"])
        self.assertRegex(
            packaged["scan_results"][0]["failure_reason"],
            r"(?is)ambiguous|conflicting",
            "A scan whose duplicated QR payloads disagree should stay explicitly ambiguous "
            "at the ingest layer instead of falling through to an unrelated unmatched path.",
        )

    def test_ingest_mc_scans_treats_lowercase_ambiguous_error_as_ambiguous(self) -> None:
        ingest_module = __import__("auto_grader.mc_scan_ingest", fromlist=["ingest_mc_scans"])
        artifact = _build_artifact()
        blank_scan = np.full((600, 900, 3), 255, dtype=np.uint8)

        with mock.patch.object(
            ingest_module,
            "read_page_identity_qr_payload",
            side_effect=ValueError("ambiguous duplicated payloads"),
        ):
            packaged = ingest_module.ingest_mc_scans({"conflicting.png": blank_scan}, artifact)

        self.assertEqual([result["status"] for result in packaged["scan_results"]], ["ambiguous"])

    def test_ingest_mc_scans_rejects_extraction_key_collisions(self) -> None:
        ingest_module = __import__("auto_grader.mc_scan_ingest", fromlist=["ingest_mc_scans"])
        artifact = _build_artifact()
        page = artifact["pages"][0]
        correct_bubble = artifact["answer_key"]["mc-1"]["correct_bubble_label"]
        matched_scan = _perspective_distort(
            _render_marked_page(page, marked_labels={"mc-1": [correct_bubble]})
        )

        with mock.patch.object(
            ingest_module,
            "extract_scored_mc_page",
            return_value={
                "scan_id": "collision",
                "page_number": page["page_number"],
                "fallback_page_code": page["fallback_page_code"],
                "normalized_image": matched_scan,
                "bubble_observations": {},
                "bubble_evidence": {},
                "marked_bubble_labels": {},
                "scored_questions": {},
            },
        ):
            with self.assertRaisesRegex(ValueError, "reserved ingest keys"):
                ingest_module.ingest_mc_scans({"matched-page.png": matched_scan}, artifact)


if __name__ == "__main__":
    unittest.main()
