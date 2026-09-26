"""Base interfaces shared by strategies (stub, no logic yet).

Planned contract: a strategy receives aligned frames (LTF with closed HTF columns
attached by :func:`smc_zero.data_loader.align_htf_to_ltf`) and returns a signal
frame indexed like the LTF input, where a signal on bar ``i`` may only use
information available at the close of bar ``i``.
"""

from __future__ import annotations

# TODO(phase-strategy): define the Signal/Position dataclasses (entry, SL, TP, reason).
# TODO(phase-strategy): define the abstract generate_signals(ltf_df, config) contract.
# TODO(tests): leak test - a signal at bar i must not change when bar i+1 changes.
