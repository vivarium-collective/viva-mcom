"""viva-mcom: process-bigraph wrapper for the MCoM microbial community model.

This package ships a **clean-room NumPy reproduction** of MCoM v1.0 (the
upstream numerical core is Julia); see ``REPRODUCTION.md`` and
``viva_mcom.engine`` for provenance and known divergences. The reproduction is
validated bit-for-bit against real MCoM output in ``datasets/golden``.
"""

from .processes import MComReproductionProcess
from .core import build_core
from .composites import mcom_coculture, mcom_seasonal

__all__ = [
    "MComReproductionProcess",
    "build_core",
    "mcom_coculture",
    "mcom_seasonal",
]
