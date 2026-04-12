from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

_SMOKE_CACHE: dict[str, object] | None = None


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


def _run_cached_smoke(test_case: unittest.TestCase) -> dict[str, object]:
    global _SMOKE_CACHE
    if _SMOKE_CACHE is None:
        run_mark_profile_smoke = _load_smoke_module(test_case)
        output_dir = Path(tempfile.mkdtemp(prefix="mc-mark-profile-smoke-contract-"))
        _SMOKE_CACHE = run_mark_profile_smoke(output_dir=output_dir)
    return _SMOKE_CACHE


class MarkProfileSmokeContractTests(unittest.TestCase):
    def test_mark_profile_smoke_reports_baseline_student_like_profiles(self) -> None:
        report = _run_cached_smoke(self)

        self.assertIsInstance(report["page_code"], str)
        self.assertTrue(report["page_code"])
        profiles = {
            profile["profile_id"]: profile
            for profile in report["profiles"]
            if profile["scan_profile_id"] == "office_scan"
        }

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
            "ambiguous_mark",
            "A very faint centered fill should now surface as review work instead "
            "of quietly vanishing into blank.",
        )
        self.assertTrue(profiles["faint_center"]["review_required"])
        self.assertEqual(
            profiles["double_mark"]["observed_status"],
            "multiple_marked",
            "The mark-profile smoke should preserve the explicit multi-mark "
            "review path instead of collapsing it into a guessed answer.",
        )
        self.assertEqual(
            profiles["hostile_correct_plus_glance"]["observed_status"],
            "correct",
            "A correct fill plus a glancing stray on another bubble should stay "
            "within the handled surface instead of escalating into fake review work.",
        )
        self.assertEqual(
            profiles["glancing_stray_only"]["observed_status"],
            "blank",
            "A lone glancing stray should stay outside the answer surface instead of "
            "pretending the student tried to fill the bubble.",
        )
        self.assertEqual(
            profiles["tiny_center_dot"]["observed_status"],
            "blank",
            "A tiny center dot should not count as an answer attempt just because it "
            "lands inside the bubble.",
        )
        self.assertEqual(
            profiles["small_check_tick"]["observed_status"],
            "blank",
            "A small check-like interior tick should still be ignored on the ordinary "
            "office-scan tier rather than forcing routine manual review.",
        )
        self.assertEqual(
            profiles["heavy_check_tick"]["observed_status"],
            "blank",
            "A heavier interior tick still should not become review work if it remains "
            "a sparse stroke instead of a real fill attempt.",
        )
        self.assertEqual(
            profiles["short_center_slash"]["observed_status"],
            "blank",
            "A short slash through the bubble should remain ignorable if we are "
            "explicitly instructing students to use strong dark fills instead.",
        )
        self.assertEqual(
            profiles["compact_center_scribble_only"]["observed_status"],
            "correct",
            "A dense compact interior scribble should cross into a real fill attempt "
            "rather than staying trapped in incidental-mark limbo.",
        )
        self.assertEqual(
            profiles["correct_plus_wrong_dot"]["observed_status"],
            "correct",
            "A plainly filled correct bubble plus a tiny accidental dot on another "
            "bubble should remain machine-gradable as the intended answer.",
        )
        self.assertEqual(
            profiles["ambiguous_patchy_center"]["observed_status"],
            "ambiguous_mark",
            "A patchy center fill near the boundary should surface as explicit "
            "review work instead of quietly becoming blank.",
        )
        self.assertTrue(profiles["ambiguous_patchy_center"]["review_required"])
        self.assertEqual(
            profiles["illegible_scratchout"]["observed_status"],
            "illegible_mark",
            "A scratchy unreadable fill should surface as illegible review work "
            "instead of pretending to be blank or confidently marked.",
        )
        self.assertTrue(profiles["illegible_scratchout"]["review_required"])
        self.assertTrue(profiles["double_mark"]["review_required"])
        self.assertFalse(profiles["solid_center"]["review_required"])

        for profile in report["profiles"]:
            image_path = Path(profile["image_path"])
            self.assertTrue(
                image_path.exists(),
                "The smoke harness should save each rendered specimen so humans can "
                "eyeball the same profiles the machine just classified.",
            )
            if profile["normalized_image_path"] is not None:
                normalized_image_path = Path(profile["normalized_image_path"])
                self.assertTrue(
                    normalized_image_path.exists(),
                    "The harness should also save the normalized readback surface so "
                    "we can inspect where the CV pipeline actually landed.",
                )

            if profile["scan_profile_id"] == "office_scan":
                self.assertEqual(
                    profile["decoded_payload"],
                    report["page_code"],
                    "The ordinary office-style scan should still decode back to the "
                    "same page identity, or the capability readout is flattering a "
                    "surface that fails under boring institutional conditions.",
                )

    def test_mark_profile_smoke_reports_realistic_scan_boundary_by_behavior_band(self) -> None:
        report = _run_cached_smoke(self)

        profile_matrix = {
            (profile["profile_id"], profile["scan_profile_id"]): profile
            for profile in report["profiles"]
        }
        scan_summaries = {
            summary["scan_profile_id"]: summary for summary in report["scan_profile_summaries"]
        }

        self.assertEqual(
            scan_summaries["clean_scan"]["severity_rank"],
            0,
            "The smoke should expose an ordered scan-profile ladder so we can talk "
            "about the strongest handled case and the next harsher failure honestly.",
        )
        self.assertEqual(scan_summaries["office_scan"]["severity_rank"], 1)
        self.assertEqual(scan_summaries["stressed_scan"]["severity_rank"], 2)

        self.assertEqual(
            profile_matrix[("solid_center", "office_scan")]["observed_behavior_band"],
            "grade",
            "A realistic office-style scan should still let a plainly filled bubble "
            "stay inside the machine-gradable surface.",
        )
        self.assertEqual(
            profile_matrix[("solid_center", "office_scan")]["expected_behavior_band"],
            "grade",
        )
        self.assertEqual(
            profile_matrix[("edge_smudge", "office_scan")]["observed_behavior_band"],
            "ignore",
            "A realistic office-style scan should still keep an edge-only smudge "
            "out of the answer surface.",
        )
        self.assertEqual(
            profile_matrix[("double_mark", "office_scan")]["observed_behavior_band"],
            "review",
            "The matrix should preserve review work under office-like scan stress "
            "instead of collapsing double marks into a guessed grade.",
        )

        boundary = report["practical_boundary"]
        self.assertIn(
            boundary["strongest_all_expected_behavior_scan_profile_id"],
            scan_summaries,
            "The smoke should summarize the strongest scan profile where the "
            "expected behavior bands still all hold.",
        )
        if boundary["next_scan_profile_with_unexpected_cases"] is not None:
            self.assertIn(
                boundary["next_scan_profile_with_unexpected_cases"],
                scan_summaries,
                "If the practical boundary has already broken, the first harsher "
                "failing profile should be named explicitly rather than left as an "
                "implied cliff in the image directory.",
            )

    def test_mark_profile_smoke_reports_incidental_mark_boundary_for_office_scan(self) -> None:
        report = _run_cached_smoke(self)

        boundary = report["incidental_mark_boundary"]
        self.assertEqual(
            boundary["scan_profile_id"],
            "office_scan",
            "The incidental-mark boundary should be reported on the realistic office "
            "scan tier, not on an unrealistically clean or already-broken stress tier.",
        )
        self.assertTrue(
            boundary["ordered_profile_ids"],
            "The boundary report should expose the ordered incidental pathology ladder "
            "instead of forcing humans to infer it from arbitrary filenames.",
        )
        self.assertIn(
            boundary["strongest_ignored_profile_id"],
            boundary["ordered_profile_ids"],
            "The strongest ignored incidental specimen should be named explicitly.",
        )
        self.assertIn(
            boundary["first_non_ignored_profile_id"],
            boundary["ordered_profile_ids"],
            "The first specimen that stops being ignored should be named explicitly.",
        )
        self.assertEqual(
            boundary["strongest_ignored_profile_id"],
            "short_center_slash",
            "The incidental-mark line should move far enough that small ticks and short "
            "slashes are still safely ignored on the ordinary office-scan tier.",
        )
        self.assertEqual(
            boundary["first_non_ignored_profile_id"],
            "compact_center_scribble_only",
            "The first non-ignored specimen should be the one that finally looks like "
            "an actual compact fill attempt, not a small incidental stroke.",
        )
        self.assertTrue(
            Path(boundary["strongest_ignored_image_path"]).exists(),
            "The strongest ignored specimen should have a saved artifact for human eyeballing.",
        )
        self.assertTrue(
            Path(boundary["first_non_ignored_image_path"]).exists(),
            "The first non-ignored specimen should have a saved artifact for human eyeballing.",
        )


if __name__ == "__main__":
    unittest.main()
