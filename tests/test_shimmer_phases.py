"""Contract tests for the coupled-oscillator shimmer phase state.

The Project Paint Dry chyron shimmer renders a stack of "layers" (top
N history lines), each pulsing with a moving highlight sweep. Prior
to coupled phases, every layer shared one period and a fixed phase
offset relative to the layer above — stable but visually frozen.

This module tests the next iteration: each layer has a SLIGHTLY
different period, and a weak Kuramoto-style coupling pulls them
toward a common phase reference so the inter-layer drift is BOUNDED.
The visible effect is a subtle "orbiting" — never in sync, never
out of sync.

The contract these tests pin down is the math, in isolation from the
rich rendering loop. The renderer's job is just to ask the state
"what phase is layer i at right now" and draw accordingly.
"""

from __future__ import annotations

import unittest


# 30 fps render loop matches the narrator reader's actual cadence.
_FRAME_DT = 1.0 / 30.0
# Six minutes of synthetic wallclock — comfortably exceeds the
# attractor's "5 minutes" satisfaction window.
_SOAK_FRAMES = int(30 * 60 * 6)
# Default base cycle and layer offset match scripts/narrator_reader.py.
_BASE_CYCLE_S = 2.7
_LAYER_OFFSET = -0.04
_NUM_LAYERS = 6


def _circular_distance(a: float, b: float) -> float:
    """Shortest distance between two phases in [0, 1) under wrap."""
    diff = abs(a - b) % 1.0
    return min(diff, 1.0 - diff)


def _max_offset_deviation(state) -> float:
    """Largest deviation of inter-layer offset from the ideal stack.

    The ideal configuration is layer i sitting at phase
    (layer 0 phase) + i * layer_offset. We measure each layer's
    actual offset relative to layer 0 and report the worst circular
    miss against the ideal.
    """
    phases = [state.phase(i) for i in range(state.num_layers)]
    p0 = phases[0]
    worst = 0.0
    for i in range(state.num_layers):
        actual_rel = (phases[i] - p0) % 1.0
        ideal_rel = (i * state.layer_offset) % 1.0
        worst = max(worst, _circular_distance(actual_rel, ideal_rel))
    return worst


def _advance_for(state, frames: int) -> None:
    for _ in range(frames):
        state.advance(_FRAME_DT)


def _make_state(coupling_strength: float | None = None):
    from auto_grader.shimmer_phases import ShimmerPhaseState

    kwargs = dict(
        num_layers=_NUM_LAYERS,
        base_cycle_s=_BASE_CYCLE_S,
        layer_offset=_LAYER_OFFSET,
    )
    if coupling_strength is not None:
        kwargs["coupling_strength"] = coupling_strength
    return ShimmerPhaseState(**kwargs)


class ShimmerPhaseStateContract(unittest.TestCase):
    def test_initial_state_is_at_ideal_offsets(self) -> None:
        """A freshly constructed state lands exactly on the ideal stack."""
        state = _make_state()
        self.assertEqual(state.num_layers, _NUM_LAYERS)
        self.assertEqual(state.layer_offset, _LAYER_OFFSET)
        # Layer 0 starts at 0.0, each successive layer at i*offset (mod 1).
        for i in range(_NUM_LAYERS):
            self.assertAlmostEqual(
                state.phase(i), (i * _LAYER_OFFSET) % 1.0, places=12
            )
        self.assertAlmostEqual(_max_offset_deviation(state), 0.0, places=12)

    def test_layers_have_distinct_natural_periods(self) -> None:
        """Each layer must have its OWN period — otherwise there's no
        drift to bound and the whole exercise is moot.

        The perturbations should be small (within ±5% of base) and
        the base cycle must remain the canonical anchor (the mean of
        the per-layer periods sits at the base cycle to within
        rounding)."""
        state = _make_state()
        periods = [state.period(i) for i in range(_NUM_LAYERS)]
        self.assertEqual(
            len(set(periods)),
            _NUM_LAYERS,
            f"layers must have distinct periods, got {periods}",
        )
        for p in periods:
            self.assertGreaterEqual(p, 0.95 * _BASE_CYCLE_S)
            self.assertLessEqual(p, 1.05 * _BASE_CYCLE_S)
        mean_period = sum(periods) / len(periods)
        self.assertAlmostEqual(mean_period, _BASE_CYCLE_S, delta=0.01)

    def test_phases_advance_over_time(self) -> None:
        """Sanity: a single second of advance should rotate layer 0
        by roughly (1 / base_cycle) of a turn."""
        state = _make_state()
        p_before = state.phase(0)
        _advance_for(state, int(round(1.0 / _FRAME_DT)))
        p_after = state.phase(0)
        expected_step = 1.0 / _BASE_CYCLE_S  # ≈ 0.370
        actual_step = (p_after - p_before) % 1.0
        # Loose tolerance — coupling and per-layer perturbation can
        # shift this slightly, but layer 0's natural period is close
        # to base.
        self.assertLess(_circular_distance(actual_step, expected_step), 0.05)

    def test_uncoupled_state_drifts_unboundedly(self) -> None:
        """Control: with coupling disabled, distinct per-layer periods
        cause the inter-layer offsets to smear far beyond the bound.
        Pins falsifiability of the coupled case — if THIS test ever
        starts passing, the bounded test below would mean nothing."""
        state = _make_state(coupling_strength=0.0)
        _advance_for(state, _SOAK_FRAMES)
        drift = _max_offset_deviation(state)
        self.assertGreater(
            drift,
            0.20,
            f"uncoupled drift was {drift:.3f}; expected unbounded "
            "smear (>0.20). If this fails, the per-layer periods "
            "are too close to identical to falsify the coupled bound.",
        )

    def test_coupled_state_drift_is_bounded_over_six_minutes(self) -> None:
        """Main contract: with weak coupling, after 6 minutes of
        synthetic wallclock, the worst inter-layer offset deviation
        from the ideal stack stays under 0.15 of a cycle."""
        state = _make_state()
        _advance_for(state, _SOAK_FRAMES)
        drift = _max_offset_deviation(state)
        self.assertLess(
            drift,
            0.15,
            f"coupled drift was {drift:.3f}, exceeds 0.15 bound",
        )

    def test_coupled_state_actually_moves(self) -> None:
        """Coupling must not be so strong that it freezes the layers
        onto the ideal stack — the whole point is visible drift."""
        state = _make_state()
        _advance_for(state, int(30 / _FRAME_DT))  # warmup
        peak = 0.0
        for _ in range(int(30 / _FRAME_DT)):
            state.advance(_FRAME_DT)
            peak = max(peak, _max_offset_deviation(state))
        self.assertGreater(
            peak,
            0.005,
            f"coupled drift peak was {peak:.4f}; coupling is so "
            "tight the layers are visually frozen — defeats purpose",
        )

    def test_drift_is_still_bounded_over_thirty_minutes(self) -> None:
        """Long-soak: bounded means bounded, not 'bounded for the
        duration of the short test.'"""
        state = _make_state()
        _advance_for(state, int(30 * 60 * 30))  # 30 min @ 30 fps
        drift = _max_offset_deviation(state)
        self.assertLess(
            drift,
            0.15,
            f"30-minute coupled drift was {drift:.3f}, exceeds 0.15",
        )

    def test_drift_bounded_at_production_layer_count(self) -> None:
        """The narrator reader instantiates ShimmerPhaseState sized to
        _VISIBLE_HISTORY_LINES (30 as of the current narrator). The
        Kuramoto bound is governed by K vs period spread, NOT by layer
        count, so the same drift bound should hold at 30 layers as at
        6. Pin that explicitly so a future bump to _VISIBLE_HISTORY_LINES
        can't silently regress the visual contract.
        """
        from auto_grader.shimmer_phases import ShimmerPhaseState

        state = ShimmerPhaseState(
            num_layers=30,
            base_cycle_s=_BASE_CYCLE_S,
            layer_offset=_LAYER_OFFSET,
        )
        _advance_for(state, _SOAK_FRAMES)
        drift = _max_offset_deviation(state)
        self.assertLess(
            drift,
            0.15,
            f"30-layer coupled drift was {drift:.3f}, exceeds 0.15",
        )

    def test_phase_query_is_a_pure_lookup(self) -> None:
        """phase(i) must not advance state — the renderer calls it
        many times per frame (once per rendered line) and frame
        advancement happens via advance() exactly once per frame."""
        state = _make_state()
        _advance_for(state, 100)
        snapshot = [state.phase(i) for i in range(_NUM_LAYERS)]
        for _ in range(50):
            for i in range(_NUM_LAYERS):
                state.phase(i)
        for i in range(_NUM_LAYERS):
            self.assertEqual(state.phase(i), snapshot[i])


if __name__ == "__main__":
    unittest.main()
