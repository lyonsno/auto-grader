"""Alt-screen contract for the Paint Dry live reader (Crispy Drips).

The Crispy Drips attractor makes the live history pane the canonical
surface for inspecting earlier in-pane history. That surface has to
OWN the viewport the operator looks at while the reader is alive:
scroll keys (k / j / u / d / 0) are the only way to see earlier
content, and the terminal's native scrollback must not accumulate
ghost frames of the Live() redraw on top of the live surface.

Satisfying that contract requires the Live() block to run in the
terminal's alternate screen buffer (Rich's `screen=True`). Without
that flag, Rich's in-place redraw leaves prior frames in the
terminal's scrollback, and scrolling the terminal natively reveals
"live view disappeared, infinite ghost header trail below" — the
exact symptom Crispy Drips is chartered to eliminate.

This test parses `scripts/narrator_reader.py` at the AST level and
asserts that the single `Live(...)` call in the module is configured
with `screen=True`. It is a structural pin, not a runtime exercise:
Rich's alt-screen plumbing itself is untestable without a real TTY.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


def _find_live_call() -> ast.Call:
    src_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "narrator_reader.py"
    )
    tree = ast.parse(src_path.read_text())
    live_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "Live":
                live_calls.append(node)
    assert live_calls, (
        "no Live(...) call found in scripts/narrator_reader.py — the "
        "alt-screen contract cannot be evaluated"
    )
    assert len(live_calls) == 1, (
        f"expected exactly one Live(...) call in narrator_reader.py, "
        f"found {len(live_calls)}"
    )
    return live_calls[0]


def _kwarg(call: ast.Call, name: str) -> ast.expr | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


class LiveAltScreenContract(unittest.TestCase):
    def test_live_block_enables_alternate_screen(self):
        call = _find_live_call()
        screen_value = _kwarg(call, "screen")
        self.assertIsNotNone(
            screen_value,
            "Live(...) must pass screen= explicitly so the alt-screen "
            "decision is visible at the call site",
        )
        # AST literal check — Rich accepts screen=True or "auto"-ish
        # values, but we want the explicit, statically verifiable
        # True so the contract is mechanical.
        self.assertIsInstance(screen_value, ast.Constant)
        self.assertIs(
            screen_value.value, True,
            "Live() must run with screen=True so the terminal's native "
            "scrollback does not accumulate ghost frames around the "
            "live surface — Crispy Drips owns in-pane history, the "
            "terminal's scrollback must not compete with it",
        )

    def test_live_block_still_drives_manual_refresh(self):
        # Guardrail: make sure this contract pin doesn't quietly erase
        # the earlier `auto_refresh=False` decision (which the inline
        # comment explains is load-bearing for shimmer phase updates).
        call = _find_live_call()
        auto_refresh = _kwarg(call, "auto_refresh")
        self.assertIsNotNone(auto_refresh)
        self.assertIsInstance(auto_refresh, ast.Constant)
        self.assertIs(auto_refresh.value, False)


if __name__ == "__main__":
    unittest.main()
