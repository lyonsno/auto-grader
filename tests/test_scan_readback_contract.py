from __future__ import annotations

import unittest

import numpy as np
from PIL import Image
import qrcode


def _load_readback_module(test_case: unittest.TestCase):
    try:
        from auto_grader.scan_readback import read_page_identity_qr_payload
    except ModuleNotFoundError:
        test_case.fail(
            "Add `auto_grader.scan_readback.read_page_identity_qr_payload(...)` so "
            "the OpenCV lane can decode page-identity QR markers from scan images."
        )
    return read_page_identity_qr_payload


def _qr_image(payload: str) -> Image.Image:
    qr = qrcode.QRCode(
        border=4,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=6,
    )
    qr.add_data(payload)
    qr.make(fit=True)
    return qr.make_image(fill_color="black", back_color="white").convert("RGB")


def _synthetic_page(*payloads: str) -> np.ndarray:
    canvas = Image.new("RGB", (900, 600), "white")
    x_positions = (40, 620)
    for x, payload in zip(x_positions, payloads, strict=False):
        canvas.paste(_qr_image(payload), (x, 40))
    return np.array(canvas)


class ScanReadbackContractTests(unittest.TestCase):
    def test_readback_returns_payload_when_duplicate_qrs_agree(self) -> None:
        read_page_identity_qr_payload = _load_readback_module(self)

        payload = read_page_identity_qr_payload(_synthetic_page("inst_test-p1", "inst_test-p1"))

        self.assertEqual(
            payload,
            "inst_test-p1",
            "OpenCV readback should resolve a duplicated page-identity QR payload when both symbols agree.",
        )

    def test_readback_rejects_mismatched_qr_payloads_as_ambiguous(self) -> None:
        read_page_identity_qr_payload = _load_readback_module(self)

        with self.assertRaisesRegex(ValueError, "Ambiguous"):
            read_page_identity_qr_payload(_synthetic_page("inst_test-p1", "inst_test-p2"))


if __name__ == "__main__":
    unittest.main()
