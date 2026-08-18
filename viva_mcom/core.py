"""Core builder for the viva-mcom workspace.

Every ``sims/run.py`` and the dashboard call :func:`build_core` (not a bare
``allocate_core()``) so the workspace's own process is registered regardless of
how the editable install was made -- ``uv``/``hatchling`` editable installs are
not always auto-discovered by ``allocate_core()``. See
docs/conventions/discovery.md and the viva-fenics precedent.
"""

from __future__ import annotations

from process_bigraph import allocate_core

from .processes import MComReproductionProcess
from .types import register_types


def build_core(core=None):
    if core is None:
        core = allocate_core()
    register_types(core)
    core.register_link("MComReproductionProcess", MComReproductionProcess)
    return core
