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
    def test_live_block_specifies_screen_mode_explicitly(self):
        call = _find_live_call()
        screen_value = _kwarg(call, "screen")
        self.assertIsNotNone(
            screen_value,
            "Live(...) must pass screen= explicitly so the screen-mode "
            "decision is visible at the call site and cannot silently "
            "revert to Rich's default",
        )
        # The live reader owns its own in-pane history surface now,
        # and oversized preview payloads are capped before they ever
        # reach the terminal. That means the old "stay in the main
        # buffer and tolerate scrollback ghosts" compromise is stale.
        # Running in the alternate screen is the clean contract: the
        # live UI should not leak prior redraw frames into native
        # terminal scrollback when the operator nudges the terminal.
        self.assertIsInstance(screen_value, ast.Constant)
        self.assertIs(
            screen_value.value, True,
            "Live() must run with screen=True so native terminal "
            "scrollback cannot reveal ghosted Paint Dry redraw frames",
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

    def test_live_block_disables_stdout_and_stderr_redirection(self):
        call = _find_live_call()
        redirect_stdout = _kwarg(call, "redirect_stdout")
        redirect_stderr = _kwarg(call, "redirect_stderr")
        self.assertIsNotNone(
            redirect_stdout,
            "Live(...) must spell out redirect_stdout= so raw Kitty control traffic doesn't silently route through Rich's stdout proxy",
        )
        self.assertIsNotNone(
            redirect_stderr,
            "Live(...) must spell out redirect_stderr= so stderr behavior stays intentional while the live window owns the terminal",
        )
        self.assertIsInstance(redirect_stdout, ast.Constant)
        self.assertIsInstance(redirect_stderr, ast.Constant)
        self.assertIs(
            redirect_stdout.value, False,
            "Live() must keep stdout attached to the real terminal so raw Kitty transmit chunks don't collide with Rich's redirected stdout wrapper",
        )
        self.assertIs(
            redirect_stderr.value, False,
            "Live() must keep stderr attached to the real terminal so diagnostics don't route through Rich's redirected stderr wrapper",
        )


if __name__ == "__main__":
    unittest.main()
