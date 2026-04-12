from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


def _load_smoke_probe():
    path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "smoke_probe.py"
    )
    spec = importlib.util.spec_from_file_location("smoke_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SmokeProbeContract(unittest.TestCase):
    def test_dry_run_accepts_explicit_model_family_and_surfaces_it(self):
        module = _load_smoke_probe()
        stdout = io.StringIO()
        stderr = io.StringIO()

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "probe-run"
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    mock.patch.object(
                        sys,
                        "argv",
                        [
                            "smoke_probe.py",
                            "--model",
                            "Step3-VL-10B",
                            "--model-family",
                            "neutral",
                            "--pick",
                            "15-blue:fr-1",
                            "--run-dir",
                            str(run_dir),
                            "--dry-run",
                        ],
                    )
                )
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                    stderr
                ):
                    code = module.main()

        self.assertEqual(code, 0)
        self.assertIn("Family:", stdout.getvalue())
        self.assertIn("neutral", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
