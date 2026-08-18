"""Unit + integration tests for MComReproductionProcess."""

import json

import numpy as np
from process_bigraph import Composite, gather_emitter_results

from viva_mcom.core import build_core
from viva_mcom.composites.coculture import _BASE_CONFIG, mcom_coculture
from viva_mcom.processes import MComReproductionProcess


def _proc(core):
    cfg = json.loads(json.dumps(_BASE_CONFIG))
    return MComReproductionProcess(
        config={"config_json": json.dumps(cfg), "dt": cfg["run"]["dt"]}, core=core)


def test_ports_are_dicts():
    core = build_core()
    p = _proc(core)
    assert isinstance(p.inputs(), dict)
    assert isinstance(p.outputs(), dict)
    assert set(p.outputs()) >= {"phytoplankton", "heterotrophs", "DIN", "DOC", "DON"}


def test_initial_state_matches_config():
    core = build_core()
    p = _proc(core)
    st = p.initial_state()
    assert st["phytoplankton"]["P0"] == _BASE_CONFIG["init"]["P_init"]
    assert st["DIN"] == _BASE_CONFIG["init"]["N_init"]
    assert st["carbon_fixed"] == 0.0


def test_single_update_returns_declared_ports():
    core = build_core()
    p = _proc(core)
    out = p.update({"irradiance": None, "nutrient_supply": None, "temperature": None},
                   interval=0.1)
    assert set(out) == set(p.outputs())
    assert isinstance(out["phytoplankton"], dict) and "P0" in out["phytoplankton"]
    assert np.isfinite(out["DIN"])


def test_irradiance_forcing_changes_growth():
    """A sibling writing a higher irradiance should increase phytoplankton growth."""
    core = build_core()
    dark = _proc(core).update(
        {"irradiance": 0.0, "nutrient_supply": None, "temperature": None}, 0.5)
    bright = _proc(core).update(
        {"irradiance": 20.0, "nutrient_supply": None, "temperature": None}, 0.5)
    assert bright["phytoplankton"]["P0"] > dark["phytoplankton"]["P0"]


def test_composite_runs_and_couples():
    """End-to-end: the composite runs and shows phototroph->heterotroph coupling."""
    core = build_core()
    doc = mcom_coculture(core, t_end=80.0)
    sim = Composite({"state": doc}, core=core)
    sim.run(80.0)
    rows = gather_emitter_results(sim)[("emitter",)]
    assert len(rows) > 10
    P = np.array([r["phytoplankton"]["P0"] for r in rows])
    H = np.array([r["heterotrophs"]["H0"] for r in rows])
    DIN = np.array([r["DIN"] for r in rows])
    # Phytoplankton blooms above its start, the heterotroph grows on the exudate,
    # and inorganic nitrogen is drawn down.
    assert P.max() > 5 * P[0]
    assert H[-1] > H[0]
    assert DIN[-1] < DIN[0]
