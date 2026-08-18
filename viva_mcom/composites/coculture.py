"""MCoM phototroph-heterotroph co-culture composites (reproduction).

All generators here are tagged **(reproduction)**: the community dynamics are
computed by the clean-room NumPy reproduction in :mod:`viva_mcom.engine`, not by
the upstream Julia engine.

The parameter set is the shipped MCoM ``configs/test.toml`` example -- a single
phytoplankton population feeding a single heterotroph through a labile + a
recalcitrant DOC/DON pool. It is fully deterministic, so the reproduction matches
upstream MCoM exactly (see ``datasets/golden`` and ``tests/test_golden.py``).
"""

from __future__ import annotations

import copy
import json

from process_bigraph.composite_generator import composite_generator

# --------------------------------------------------------------------------- #
# Base config: MCoM configs/test.toml (1P / 1H batch co-culture)
# --------------------------------------------------------------------------- #

_BASE_CONFIG = {
    "run": {"runid": "coculture", "system_seed": 42, "dynamics_seed": 100,
            "t_end": 200.0, "dt": 0.005, "dt_out": 0.1},
    "variant": {"use_PI_curve": True, "use_exudation": False,
                "surplus_remineralization": True, "competing_pairs": False},
    "dims": {"P": 1, "H": 1, "DOC": 2, "DON": 2, "M": 1},
    "network": {"explicit": True,
                "exPDOC": [[1, 2]], "exPDON": [[1, 2]],
                "exHDOC": [[1, 2]], "exHDON": [[1, 2]],
                "upDOCH": [[1]], "upDONH": [[1]],
                "prodHM": [[]], "effsMP": [[]]},
    "init": {"t_init": 0.0, "N_init": 8.8, "P_init": 0.01, "DOC_init": 1.0,
             "DON_init": 0.2, "H_init": 0.003, "M_init": 0.0},
    "environment": {"irradiance": "constant(5)", "D": 0.0, "d_M": 0.1,
                    "SourceNrnd": 0.0, "nutrient": "constant(0)"},
    "dynamics": {
        "explicit": ["R_HDON", "R_HDOC", "R_PDON", "R_PDOC"],
        "h_M": 0.00001, "V_NP": 1.7, "XP_C": 12.0, "K_NP": 2.0, "d_P": 0.28,
        "d_P2": 0.0, "lambda": 0.0, "rCN_P": 5.2, "rChlC": 1.0, "q_ex": 0.15,
        "rCN_ex": 3.0, "I_max": 5.0, "I_alpha": 0.08, "I_beta": 0.003,
        "V_NH": 4.0, "K_NH": 0.1, "XH_C": 12.0, "d_H": 0.2, "d_H2": 0.001,
        "rCN_H": 4.0, "Y_DOC": 0.87, "Y_DON": 0.95, "K_DOCH": 0.1, "K_DONH": 0.1,
        "V_DOCH": 12.2, "V_DONH": 5.2,
        "R_HDOC": [[0.99, 0.01]], "R_HDON": [[0.93, 0.07]],
        "R_PDOC": [[1.0, 0.0]], "R_PDON": [[1.0, 0.0]],
        "P_HM": 0.00001, "A_MP": 1.0,
    },
}

_EMIT = {
    "phytoplankton": "map[string,float]", "heterotrophs": "map[string,float]",
    "DOC": "map[string,float]", "DON": "map[string,float]",
    "metabolites": "map[string,float]", "DIN": "float",
    "carbon_fixed": "float", "carbon_lost": "float",
    "P_total": "float", "H_total": "float", "global_time": "float",
}


def _document(cfg, interval):
    """Wire the MComReproductionProcess + RAMEmitter into a bigraph document."""
    community = ("phytoplankton", "heterotrophs", "DOC", "DON", "metabolites",
                 "DIN", "carbon_fixed", "carbon_lost", "P_total", "H_total")
    return {
        "mcom": {
            "_type": "process",
            "address": "local:MComReproductionProcess",
            "config": {"config_json": json.dumps(cfg), "dt": cfg["run"]["dt"]},
            "interval": interval,
            "inputs": {
                "irradiance": ["stores", "irradiance"],
                "nutrient_supply": ["stores", "nutrient_supply"],
                "temperature": ["stores", "temperature"],
            },
            "outputs": {k: ["stores", k] for k in community},
        },
        # Forcing stores default to None -> the config's own time-functions are
        # used (exact standalone MCoM). A sibling process may write these to
        # drive the community (the composable coupling surface).
        "stores": {"irradiance": None, "nutrient_supply": None, "temperature": None},
        "emitter": {
            "_type": "step",
            "address": "local:RAMEmitter",
            "config": {"emit": _EMIT},
            "inputs": {**{k: ["stores", k] for k in community},
                       "global_time": ["global_time"]},
        },
    }


@composite_generator(
    name="mcom_coculture",
    description="(reproduction) MCoM phototroph-heterotroph batch co-culture: a "
                "phytoplankton bloom fuels a heterotroph via exuded DOC/DON while "
                "inorganic nitrogen draws down.",
    parameters={
        "irradiance": {"type": "float", "default": 5.0,
                       "description": "Constant light intensity"},
        "N_init": {"type": "float", "default": 8.8,
                   "description": "Initial inorganic nitrogen [mM N]"},
        "P_init": {"type": "float", "default": 0.01,
                   "description": "Initial phytoplankton density [1e6 cells/mL]"},
        "H_init": {"type": "float", "default": 0.003,
                   "description": "Initial heterotroph density [1e6 cells/mL]"},
        "t_end": {"type": "float", "default": 200.0,
                  "description": "Simulation length [days]"},
    },
)
def mcom_coculture(core=None, *, irradiance=5.0, N_init=8.8, P_init=0.01,
                   H_init=0.003, t_end=200.0):
    cfg = copy.deepcopy(_BASE_CONFIG)
    cfg["run"]["runid"] = "coculture"
    cfg["run"]["t_end"] = float(t_end)
    cfg["environment"]["irradiance"] = f"constant({irradiance})"
    cfg["init"]["N_init"] = float(N_init)
    cfg["init"]["P_init"] = float(P_init)
    cfg["init"]["H_init"] = float(H_init)
    return _document(cfg, interval=cfg["run"]["dt_out"])


@composite_generator(
    name="mcom_seasonal",
    description="(reproduction) MCoM community in a seasonally-lit chemostat: a "
                "steady nutrient inflow with sinusoidal (annual) irradiance drives "
                "a sustained phytoplankton-heterotroph community that blooms and "
                "recedes with the light cycle.",
    parameters={
        "nutrient_supply": {"type": "float", "default": 10.0,
                            "description": "Constant inflowing DIN concentration N_ext [mM N]"},
        "irradiance_mean": {"type": "float", "default": 5.0,
                            "description": "Baseline (annual-mean) irradiance"},
        "irradiance_amplitude": {"type": "float", "default": 6.0,
                                 "description": "Seasonal irradiance amplitude"},
        "dilution_rate": {"type": "float", "default": 0.1,
                          "description": "Chemostat dilution/exchange rate D [1/day]"},
        "t_end": {"type": "float", "default": 1460.0,
                  "description": "Simulation length [days] (>=4 seasonal cycles)"},
    },
)
def mcom_seasonal(core=None, *, nutrient_supply=10.0, irradiance_mean=5.0,
                  irradiance_amplitude=6.0, dilution_rate=0.1, t_end=1460.0):
    cfg = copy.deepcopy(_BASE_CONFIG)
    cfg["run"]["runid"] = "seasonal"
    cfg["run"]["t_end"] = float(t_end)
    # Seasonal light: baseline + annual sinusoid (period 365 d, peak mid-year).
    cfg["environment"]["irradiance"] = (
        f"{irradiance_mean} + sinus({irradiance_amplitude}, 365, 182.5)")
    # Steady nutrient inflow delivered through the dilution term D*(N_ext - N).
    cfg["environment"]["nutrient"] = f"constant({nutrient_supply})"
    cfg["environment"]["D"] = float(dilution_rate)
    cfg["init"]["N_init"] = float(nutrient_supply)
    return _document(cfg, interval=cfg["run"]["dt_out"])
