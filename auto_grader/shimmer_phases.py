"""Coupled-oscillator phase state for the Project Paint Dry shimmer.

The narrator reader's chyron renders a stack of "layers" (top N
history lines), each pulsing with a moving highlight sweep. Before
this module existed, every layer shared one period and a fixed
phase offset relative to the layer above — the inter-layer phases
were rigidly locked, which read visually as "frozen."

This module gives each layer a slightly different natural period
and weakly couples them (Kuramoto-style) toward a common phase
reference. The result is bounded drift: the layers visibly orbit
around the ideal stack instead of marching in lockstep, but they
never smear into uncorrelated noise.

Math notes
----------
Phases are stored in [0, 1) (one full cycle = 1.0). The "ideal
stack" places layer i at phase ``layer_index * layer_offset``
relative to layer 0. To apply Kuramoto coupling around that ideal
configuration we work in CORRECTED coordinates ``φ_i = θ_i - i·offset``
— in those coordinates the ideal is "all layers at the same phase"
which is exactly the standard Kuramoto sync target. The coupling
term pulls each corrected phase toward the circular mean of the
others, so the ideal stack is restored without freezing motion.

The class is rendering-agnostic stdlib math; the renderer in
``scripts/narrator_reader.py`` constructs one instance and asks it
for ``phase(layer_index)`` each frame.
"""

from __future__ import annotations

import math
from typing import List


# Default ±1.5% spread on per-layer periods. Small enough that the
# inter-layer beat is subtle (visible drift over seconds, not a
# strobing flicker), large enough that the uncoupled control test
# blows past the 0.20-cycle "smear" threshold within minutes.
_DEFAULT_PERIOD_SPREAD = 0.015

# Coupling strength K is in units of "phase units pulled per second
# per unit of phase error." With ±1.5% period spread, the steady-state
# worst-case offset is roughly (2 * spread / base_cycle) / K — at K=0.10
# that lands well under the 0.15-cycle drift bound while still leaving
# visible orbiting. Empirically tuned by the contract test suite.
_DEFAULT_COUPLING_STRENGTH = 0.10


def _layer_period_perturbations(num_layers: int, spread: float) -> List[float]:
    """Return ``num_layers`` multiplicative perturbations centered on 1.

    The perturbations are spread symmetrically around 1.0 in the
    range ``[1 - spread, 1 + spread]`` so that their arithmetic
    mean is exactly 1.0 — this keeps the base cycle as the
    canonical anchor (one of the attractor's constraints).

    For ``num_layers == 1`` we return ``[1.0]`` (degenerate, no
    spread to apply).
    """
    if num_layers <= 1:
        return [1.0]
    # Linear ramp from -spread to +spread, inclusive on both ends.
    step = (2 * spread) / (num_layers - 1)
    return [1.0 - spread + i * step for i in range(num_layers)]


class ShimmerPhaseState:
    """Per-layer phase state with weak Kuramoto coupling.

    Parameters
    ----------
    num_layers:
        Number of independent shimmer layers to track. The renderer
        passes layer indices in ``[0, num_layers)``; queries with
        an out-of-range index are clamped to the last layer (which
        matches the renderer's existing ``min(layer_index,
        MAX_LAYERS - 1)`` clamping).
    base_cycle_s:
        The canonical period in seconds. Per-layer perturbations
        sit symmetrically around this value so the mean per-layer
        period equals ``base_cycle_s`` exactly.
    layer_offset:
        The intended fixed phase offset (in cycles) between
        adjacent layers. The Kuramoto coupling restores this
        configuration when drift would otherwise smear it out.
    coupling_strength:
        Kuramoto K in cycles-per-second-per-unit-error. Set to
        ``0.0`` for an uncoupled control. Default is gentle enough
        to leave visible orbiting.
    period_spread:
        Half-width of the per-layer period perturbation, as a
        fraction of ``base_cycle_s``. Default ±1.5%.
    """

    def __init__(
        self,
        num_layers: int,
        base_cycle_s: float,
        layer_offset: float,
        coupling_strength: float = _DEFAULT_COUPLING_STRENGTH,
        period_spread: float = _DEFAULT_PERIOD_SPREAD,
    ) -> None:
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        if base_cycle_s <= 0:
            raise ValueError("base_cycle_s must be positive")
        self.num_layers = num_layers
        self.base_cycle_s = base_cycle_s
        self.layer_offset = layer_offset
        self.coupling_strength = coupling_strength
        perturbations = _layer_period_perturbations(num_layers, period_spread)
        self._periods: List[float] = [
            base_cycle_s * p for p in perturbations
        ]
        # Initialize each layer at the ideal stack so the very first
        # frame doesn't show a coupling transient.
        self._phases: List[float] = [
            (i * layer_offset) % 1.0 for i in range(num_layers)
        ]

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def phase(self, layer_index: int) -> float:
        """Current phase of layer ``layer_index`` in [0, 1).

        Pure lookup — does not advance state. Indices past the last
        layer are clamped, matching the renderer's MAX_LAYERS cap.
        """
        if layer_index < 0:
            layer_index = 0
        elif layer_index >= self.num_layers:
            layer_index = self.num_layers - 1
        return self._phases[layer_index]

    def period(self, layer_index: int) -> float:
        """Natural period (seconds) of layer ``layer_index``."""
        if layer_index < 0:
            layer_index = 0
        elif layer_index >= self.num_layers:
            layer_index = self.num_layers - 1
        return self._periods[layer_index]

    # ------------------------------------------------------------------
    # Update step
    # ------------------------------------------------------------------

    def advance(self, dt: float) -> None:
        """Advance every layer by ``dt`` seconds of natural rotation
        plus one Kuramoto coupling step.

        Called once per render frame from the narrator reader; cheap
        enough to run at 30 fps with a handful of layers and pure
        Python arithmetic.
        """
        if dt <= 0:
            return

        # Step 1: free natural rotation. Each layer turns at its own
        # rate (1 / period_i cycles per second).
        new_phases = [
            (p + dt / period) % 1.0
            for p, period in zip(self._phases, self._periods)
        ]

        # Step 2: Kuramoto pull toward the ideal stack. We work in
        # corrected coordinates φ_i = θ_i - i·offset so the ideal
        # is "all corrected phases equal" — the standard Kuramoto
        # sync target.
        n = self.num_layers
        if n > 1 and self.coupling_strength > 0:
            two_pi = 2.0 * math.pi
            corrected_x = 0.0
            corrected_y = 0.0
            for i, p in enumerate(new_phases):
                corrected = (p - i * self.layer_offset) % 1.0
                angle = corrected * two_pi
                corrected_x += math.cos(angle)
                corrected_y += math.sin(angle)
            # Circular mean of corrected phases. atan2 returns
            # radians in (-π, π]; convert back to cycles in [0, 1).
            mean_corrected = (
                math.atan2(corrected_y, corrected_x) / two_pi
            ) % 1.0

            k_dt = self.coupling_strength * dt
            for i in range(n):
                target = (mean_corrected + i * self.layer_offset) % 1.0
                # Shortest signed circular distance from current to
                # target, in [-0.5, 0.5).
                delta = ((target - new_phases[i]) + 0.5) % 1.0 - 0.5
                new_phases[i] = (new_phases[i] + k_dt * delta) % 1.0

        self._phases = new_phases
