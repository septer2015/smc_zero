"""The SMC M15 entry strategy (stub, no logic yet).

Planned contract (constitution "[Иерархия ТФ]"): D1 supplies the global bias,
PDH/PDL and major order blocks; H1 supplies the working structure, premium /
discount and sessions; M15 supplies the entry - sweep -> CHoCH -> entry into an
M15 FVG/OB that sits inside the H1 zone.  Every input row must be closed
(``is_closed`` True) and HTF columns must come from
:func:`smc_zero.data_loader.align_htf_to_ltf`, never from a raw HTF join.
"""

from __future__ import annotations

# TODO(phase-strategy): HTF bias filter from closed D1/H1 structure.
# TODO(phase-strategy): premium/discount gate (long in discount, short in premium).
# TODO(phase-strategy): M15 sweep -> CHoCH trigger with entry inside H1 FVG/OB.
# TODO(tests): synthetic test per rule + leak test on the HTF stitched columns.
