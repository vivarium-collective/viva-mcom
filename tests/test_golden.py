"""Fidelity tests: the reproduction must match real MCoM Julia output.

``datasets/golden/{test_tinit,test_save_offset}`` contain trajectories produced
by the *upstream* MCoM v1.0 Julia engine (shipped in the MCoM repo's test suite)
together with the exact config that generated them. Because those configs are
deterministic (``SourceNrnd = 0``, scalar/degenerate-range/explicit parameters),
the clean-room reproduction reproduces them bit-for-bit.
"""

from pathlib import Path

import numpy as np
import pytest
import toml

from viva_mcom.config import instantiate_params
from viva_mcom.engine import MComEngine

GOLDEN = Path(__file__).resolve().parents[1] / "datasets" / "golden"


@pytest.mark.parametrize("case", ["test_tinit", "test_save_offset"])
def test_reproduction_matches_upstream(case):
    cfg = toml.load(GOLDEN / case / "mcom_config.toml")
    run = cfg["run"]
    params = instantiate_params(cfg)
    eng = MComEngine(params, dt=run["dt"])
    traj = eng.run(run["t_end"], run["dt_out"], save_offset=run.get("save_offset", 0.0))

    def gold(var):
        return np.loadtxt(GOLDEN / case / f"data_{var}.mcom")

    g_t, g_P, g_N = gold("t"), gold("P"), gold("N")
    n = len(g_t)
    assert len(traj["t"]) == n, "sample count differs from upstream"

    # Upstream writes 36-digit floats; the reproduction is exact to ~1e-9.
    assert np.max(np.abs(traj["t"] - g_t)) < 1e-9
    assert np.max(np.abs(traj["P"][:, 0] - g_P)) < 1e-9
    assert np.max(np.abs(traj["N"] - g_N)) < 1e-9
