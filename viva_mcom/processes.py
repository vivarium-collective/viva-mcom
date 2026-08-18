"""Process-bigraph wrapper for the MCoM microbial community model.

``MComReproductionProcess`` drives the clean-room NumPy reproduction in
:mod:`viva_mcom.engine` -- it is **not** a bridge to the upstream Julia engine
(see the module docstring of :mod:`viva_mcom.engine` and ``REPRODUCTION.md`` for
the provenance and known divergences). The bare name ``MComProcess`` is
deliberately reserved for a future real Julia bridge.

Design
------
The process owns the authoritative community state (all phytoplankton,
heterotroph, DOC, DON, metabolite, and inorganic-nitrogen variables) and steps
it with MCoM's own 2nd-order Adams-Bashforth integrator.

Inputs (the composable coupling surface -- a sibling environment/forcing process
can write any of these; left unwired they fall back to the config's time
functions, i.e. exact standalone MCoM):

    irradiance       maybe[float]  light intensity forcing
    nutrient_supply  maybe[float]  external DIN concentration N_ext (enters via the
                                   D * (N_ext - N) dilution term)
    temperature      maybe[float]  (accepted; upstream temperature modulation is a stub)

Outputs (per-interval deltas, so they compose additively with sibling processes;
``map[string,float]`` entries are keyed ``P0..``, ``H0..``, ``DOC0..`` etc.):

    phytoplankton    map[string,float]   d(cell density) [1e6 cells/mL]
    heterotrophs     map[string,float]   d(cell density) [1e6 cells/mL]
    DOC              map[string,float]   d(dissolved organic carbon) [mM C]
    DON              map[string,float]   d(dissolved organic nitrogen) [mM N]
    metabolites      map[string,float]   d(metabolite level)
    DIN              float               d(inorganic nitrogen) [mM N]
    carbon_fixed     float               d(cumulative fixed carbon) [mM C]
    carbon_lost      float               d(cumulative respired/exported carbon) [mM C]
"""

from __future__ import annotations

import json

from process_bigraph import Process

from .config import instantiate_params
from .engine import MComEngine


class MComReproductionProcess(Process):
    """MCoM community dynamics (clean-room reproduction of the Julia engine)."""

    config_schema = {
        # Full MCoM TOML config (dims / variant / init / environment / dynamics
        # [/ network]) serialised as JSON, so the whole heterogeneous config
        # travels through one typed field.
        "config_json": "string",
        # Integrator substep (days). Should match the config's [run].dt.
        "dt": {"_type": "float", "_default": 0.001},
        # Seed for the DIN-supply noise (only active when SourceNrnd != 0).
        "seed": {"_type": "integer", "_default": 0},
        "source_noise": {"_type": "boolean", "_default": False},
    }

    def __init__(self, config=None, core=None):
        super().__init__(config=config, core=core)
        cfg = json.loads(self.config["config_json"])
        self._params = instantiate_params(cfg)
        self._engine: MComEngine | None = None

    # -- discovery of dynamic map keys ------------------------------------- #
    def _keys(self, prefix, n):
        return [f"{prefix}{i}" for i in range(n)]

    def _build(self):
        p = self.config
        self._engine = MComEngine(
            self._params, dt=p["dt"],
            seed=p["seed"], source_noise=p["source_noise"],
        )

    # -- bigraph interface -------------------------------------------------- #
    def inputs(self):
        return {
            "irradiance": "maybe[float]",
            "nutrient_supply": "maybe[float]",
            "temperature": "maybe[float]",
        }

    def outputs(self):
        return {
            "phytoplankton": "map[string,float]",
            "heterotrophs": "map[string,float]",
            "DOC": "map[string,float]",
            "DON": "map[string,float]",
            "metabolites": "map[string,float]",
            "DIN": "float",
            "carbon_fixed": "float",
            "carbon_lost": "float",
            # Scalar community summaries (total biomass across populations),
            # convenient targets for behaviour tests / plots.
            "P_total": "float",
            "H_total": "float",
        }

    def initial_state(self):
        p = self._params
        return {
            "phytoplankton": dict(zip(self._keys("P", p.nP), map(float, p.P_init))),
            "heterotrophs": dict(zip(self._keys("H", p.nH), map(float, p.H_init))),
            "DOC": dict(zip(self._keys("DOC", p.nDOC), map(float, p.DOC_init))),
            "DON": dict(zip(self._keys("DON", p.nDON), map(float, p.DON_init))),
            "metabolites": dict(zip(self._keys("M", p.nM), map(float, p.M_init))),
            "DIN": float(p.N_init),
            "carbon_fixed": 0.0,
            "carbon_lost": 0.0,
            "P_total": float(p.P_init.sum()),
            "H_total": float(p.H_init.sum()),
        }

    def update(self, state, interval):
        if self._engine is None:
            self._build()

        forcings = {}
        for eng_key, port in (("irradiance", "irradiance"),
                              ("nutrient", "nutrient_supply"),
                              ("temperature", "temperature")):
            v = state.get(port)
            if v is not None:
                forcings[eng_key] = float(v)

        s = self._engine.state
        before = (s.P.copy(), s.H.copy(), s.DOC.copy(), s.DON.copy(),
                  s.M.copy(), s.N, s.Cin_total, s.Cout_total)
        self._engine.advance(interval, **forcings)
        s = self._engine.state

        p = self._params

        def dmap(prefix, new, old, n):
            return {f"{prefix}{i}": float(new[i] - old[i]) for i in range(n)}

        return {
            "phytoplankton": dmap("P", s.P, before[0], p.nP),
            "heterotrophs": dmap("H", s.H, before[1], p.nH),
            "DOC": dmap("DOC", s.DOC, before[2], p.nDOC),
            "DON": dmap("DON", s.DON, before[3], p.nDON),
            "metabolites": dmap("M", s.M, before[4], p.nM),
            "DIN": float(s.N - before[5]),
            "carbon_fixed": float(s.Cin_total - before[6]),
            "carbon_lost": float(s.Cout_total - before[7]),
            "P_total": float(s.P.sum() - before[0].sum()),
            "H_total": float(s.H.sum() - before[1].sum()),
        }
