"""Regression tests against real scanner-PDF page images.

These tests use PNG fixtures rasterized at 200 DPI from the real
generated-exam scanner PDF (04122026.pdf). At this DPI, pages 3 and 5
fall in OpenCV's QR detector blind spot at native resolution and
require the multi-scale retry path to decode. See
tests/fixtures/scanner_pdf_qr_regression/PROVENANCE.md for details.

This test surface exists because synthetic blur/noise probes and
hand-picked page images can flatter one retry strategy while hiding
real scanner-produced pathologies. The acceptance criterion is that
all five pages decode without page-by-page manual rasterization
selection.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2

from auto_grader.scan_readback import read_page_identity_qr_payload

_FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "scanner_pdf_qr_regression"

# All five pages in the generated-exam demo share the same instance code.
_EXPECTED_PAYLOAD_PREFIX = "inst_"


class ScannerPdfQrRegressionTests(unittest.TestCase):
    def _load_page(self, page_num: int):
        path = _FIXTURES_DIR / f"page_{page_num}_200dpi.png"
        if not path.exists():
            self.skipTest(
                f"Scanner PDF fixture not found: {path}. "
                "See tests/fixtures/scanner_pdf_qr_regression/PROVENANCE.md."
            )
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            self.fail(f"Failed to load fixture image: {path}")
        return image

    def test_all_five_pages_decode_without_manual_scale_selection(self) -> None:
        """The satisfaction condition: the page-identity path recovers QR
        codes from all five scanner-produced pages without page-by-page
        manual rasterization selection.
        """
        for page_num in range(1, 6):
            image = self._load_page(page_num)
            payload = read_page_identity_qr_payload(image)
            self.assertTrue(
                payload.startswith(_EXPECTED_PAYLOAD_PREFIX),
                f"Page {page_num}: payload {payload!r} does not start with "
                f"{_EXPECTED_PAYLOAD_PREFIX!r}",
            )

    def test_page_3_requires_rescale_at_200_dpi(self) -> None:
        """Page 3 at 200 DPI is the canonical rescale-dependent fixture.
        Guard that native detection still fails so this test stays honest
        about exercising the rescale path.
        """
        from auto_grader.scan_readback import _detect_at_native_resolution

        image = self._load_page(3)

        self.assertEqual(
            _detect_at_native_resolution(image),
            [],
            "Test fixture precondition: page 3 at 200 DPI should fail "
            "native-resolution detection. If OpenCV improved, update "
            "PROVENANCE.md and remove this guard.",
        )

        payload = read_page_identity_qr_payload(image)
        self.assertTrue(payload.startswith(_EXPECTED_PAYLOAD_PREFIX))

    def test_page_5_requires_rescale_at_200_dpi(self) -> None:
        """Page 5 at 200 DPI requires 2.5× upscaling — the widest scale
        factor in the retry set. Guard that native detection still fails.
        """
        from auto_grader.scan_readback import _detect_at_native_resolution

        image = self._load_page(5)

        self.assertEqual(
            _detect_at_native_resolution(image),
            [],
            "Test fixture precondition: page 5 at 200 DPI should fail "
            "native-resolution detection. If OpenCV improved, update "
            "PROVENANCE.md and remove this guard.",
        )

        payload = read_page_identity_qr_payload(image)
        self.assertTrue(payload.startswith(_EXPECTED_PAYLOAD_PREFIX))


if __name__ == "__main__":
    unittest.main()
