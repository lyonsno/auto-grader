from __future__ import annotations

from pathlib import Path
import tempfile
import unittest


def _load_smoke_module(test_case: unittest.TestCase):
    try:
        from auto_grader.mark_profile_smoke import run_mark_profile_smoke
    except ModuleNotFoundError:
        test_case.fail(
            "Add `auto_grader.mark_profile_smoke.run_mark_profile_smoke(...)` so the "
            "OpenCV lane can generate a realistic mark-profile matrix instead of only "
            "isolated helper-level edge cases."
        )
    return run_mark_profile_smoke


class MarkProfileSmokeContractTests(unittest.TestCase):
    def test_mark_profile_smoke_reports_baseline_student_like_profiles(self) -> None:
        run_mark_profile_smoke = _load_smoke_module(self)

        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_mark_profile_smoke(output_dir=Path(tmpdir))

            self.assertIsInstance(report["page_code"], str)
            self.assertTrue(report["page_code"])
            profiles = {profile["profile_id"]: profile for profile in report["profiles"]}

            self.assertEqual(
                profiles["solid_center"]["observed_status"],
                "correct",
                "A solid centered fill should remain the boring baseline that scores as correct.",
            )
            self.assertEqual(
                profiles["light_center"]["observed_status"],
                "correct",
                "The lighter centered fill profile should reflect the recent blur-tolerant "
                "bubble hardening rather than regressing to blank.",
            )
            self.assertEqual(
                profiles["scribble_center"]["observed_status"],
                "correct",
                "A realistic scribbly centered fill should still read as a mark.",
            )
            self.assertEqual(
                profiles["off_center_patch"]["observed_status"],
                "correct",
                "An intentional but off-center filled patch should stay inside the "
                "handled surface rather than quietly dropping to blank.",
            )
            self.assertEqual(
                profiles["edge_smudge"]["observed_status"],
                "blank",
                "The mark-profile smoke should prove the reader still ignores "
                "edge-only smudges in a more student-like rendering path.",
            )
            self.assertEqual(
                profiles["faint_center"]["observed_status"],
                "blank",
                "The smoke should keep reporting where the current blank boundary "
                "still is instead of flattering the handled surface.",
            )
            self.assertEqual(
                profiles["double_mark"]["observed_status"],
                "multiple_marked",
                "The mark-profile smoke should preserve the explicit multi-mark "
                "review path instead of collapsing it into a guessed answer.",
            )
            self.assertTrue(profiles["double_mark"]["review_required"])
            self.assertFalse(profiles["solid_center"]["review_required"])

            for profile in report["profiles"]:
                image_path = Path(profile["image_path"])
                normalized_image_path = Path(profile["normalized_image_path"])
                self.assertTrue(
                    image_path.exists(),
                    "The smoke harness should save each rendered specimen so humans can "
                    "eyeball the same profiles the machine just classified.",
                )
                self.assertTrue(
                    normalized_image_path.exists(),
                    "The harness should also save the normalized readback surface so "
                    "we can inspect where the CV pipeline actually landed.",
                )
                self.assertEqual(
                    profile["decoded_payload"],
                    report["page_code"],
                    "Every synthetic profile should still decode back to the same page "
                    "identity after degradation, or the capability readout is lying "
                    "about the surface being exercised.",
                )


if __name__ == "__main__":
    unittest.main()
