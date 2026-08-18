"""Composite-generator registration + discovery tests."""

import numpy as np
from process_bigraph import Composite, gather_emitter_results
from process_bigraph.composite_generator import _REGISTRY

from viva_mcom.core import build_core
from viva_mcom.composites.coculture import mcom_coculture, mcom_seasonal


def test_generators_are_registered():
    ids = list(_REGISTRY)
    for gen in ("mcom_coculture", "mcom_seasonal"):
        assert any(eid.endswith(f".{gen}") or eid == gen for eid in ids), \
            f"{gen} missing from registry; have {ids[:8]}"


def test_seasonal_shows_forced_variation():
    """Under seasonal forcing the phytoplankton should vary through the year."""
    core = build_core()
    doc = mcom_seasonal(core, t_end=400.0)
    sim = Composite({"state": doc}, core=core)
    sim.run(400.0)
    rows = gather_emitter_results(sim)[("emitter",)]
    P = np.array([r["phytoplankton"]["P0"] for r in rows])
    P = P[P > 0]
    # Seasonal light drives a non-trivial swing in phytoplankton density.
    assert P.max() > 2 * np.median(P)
