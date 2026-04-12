from __future__ import annotations

import importlib
import unittest


class GeneratedExamDemoImportContract(unittest.TestCase):
    def test_module_import_does_not_require_pdf_only_dependencies(self) -> None:
        try:
            module = importlib.import_module("auto_grader.generated_exam_demo")
        except ModuleNotFoundError as exc:
            self.fail(
                "generated_exam_demo should import cleanly even when PDF-only "
                f"dependencies are absent; got {exc!r}"
            )

        self.assertTrue(
            hasattr(module, "build_generated_mc_exam_demo_packet"),
            "the generated exam demo builder should stay importable without "
            "forcing the PDF renderer dependency tree at module import time",
        )


if __name__ == "__main__":
    unittest.main()
