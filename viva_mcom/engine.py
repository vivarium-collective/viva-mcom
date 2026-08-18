"""Clean-room NumPy reproduction of the MCoM 1.0 microbial community model.

THIS IS A REPRODUCTION, NOT THE UPSTREAM ENGINE.
------------------------------------------------
This module re-implements, in portable Python/NumPy, the published algorithm of
the **Microbial Community Model (MCoM) v1.0** whose numerical core is written in
Julia. It does **not** call, embed, or wrap the upstream Julia code. It exists so
the model can run in environments without a Julia toolchain.

Reproduced from
    * Paper: L. Lücken, M. J. Follows, J. G. Bragg, S. T. Lennartz,
      "The microbial community model MCoM 1.0: a scalable framework for modelling
      phototroph-heterotrophic interactions in diverse microbial communities",
      Geosci. Model Dev., 19, 2461-2477, 2026.
      https://doi.org/10.5194/gmd-19-2461-2026
    * Source: https://github.com/bgc-mod/MCoM  (v1.0.0, commit a39227c, MIT license,
      Copyright (c) 2025 bgc-mod), Zenodo https://doi.org/10.5281/zenodo.19108362

The reproduced arithmetic follows the Julia source verbatim:
    src/mcom.jl/src/mcom_system.jl   -> derivative RHS  (see `derivatives`)
    src/mcom.jl/src/mcom_numerics.jl -> 2nd-order Adams-Bashforth step
    src/mcom.jl/src/mcom_io.jl       -> environmental fluctuation grammar
    src/mcom.jl/src/mcom_generator.jl-> parameter/network instantiation
    src/mcom.jl/main.jl              -> integration loop / save cadence

Known divergences from upstream (see also REPRODUCTION.md):
    * Random parameter draws (a ``[min, max]`` range with ``min < max``, and the
      Dirichlet/simplex splitting of un-normalized release/rate matrices across
      >1 connected compounds) are **not** reproduced, because they depend on
      Julia's Mersenne-Twister stream, which cannot be reproduced bit-for-bit in
      NumPy. Deterministic configs -- scalars, degenerate ranges ``[v, v]``,
      explicit values, and single-edge networks -- reproduce upstream exactly.
      A `NotImplementedError` is raised where an RNG-dependent draw is required.
    * Temperature modulation of rates is a stub upstream (``tempfac = 1.0``); this
      reproduction keeps the same stub.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

# --------------------------------------------------------------------------- #
# Environmental fluctuation grammar (reproduces mcom_io.jl:make_fct)
# --------------------------------------------------------------------------- #

FluctFn = Callable[[float], float]


def _make_component(env_var: str, spec: str) -> FluctFn:
    """Build one fluctuation component from a ``type(args)`` string or a number.

    Reproduces the elementary function forms of ``mcom_io.jl``:
    ``constant(v)``, ``sinus(amp[,period=365][,offset=0])``,
    ``pulse(offset,duration,amp[,period=365])``. A bare number is a constant.
    """
    spec = spec.strip()
    # Bare numeric literal -> constant
    try:
        v = float(spec)
        if v < 0:
            raise ValueError(f"'{env_var}': no negative values allowed (given {v})")
        return lambda _t, _v=v: _v
    except ValueError:
        pass

    m = re.fullmatch(r"([A-Za-z_]+)\s*\((.*)\)", spec)
    if not m:
        raise ValueError(f"Malformed function definition for '{env_var}': '{spec}'")
    fct_type = m.group(1).strip().lower()
    args = [float(a.strip()) for a in m.group(2).split(",")] if m.group(2).strip() else []

    if fct_type == "constant":
        if len(args) != 1:
            raise ValueError(f"'{env_var}': constant() needs exactly one arg (given {args})")
        v = args[0]
        if v < 0:
            raise ValueError(f"'{env_var}': constant value can't be negative (given {v})")
        return lambda _t, _v=v: _v

    if fct_type == "sinus":
        if not 1 <= len(args) <= 3:
            raise ValueError(f"'{env_var}': sinus() needs 1-3 args (given {args})")
        amp = args[0]
        period = args[1] if len(args) > 1 else 365.0
        offset = args[2] if len(args) > 2 else 0.0
        if amp < 0 or period <= 0 or period <= offset:
            raise ValueError(f"'{env_var}': invalid sinus() args {args}")
        return lambda t, a=amp, p=period, o=offset: a * (1 + math.sin(2 * math.pi * (t - o) / p)) / 2

    if fct_type == "pulse":
        if not 3 <= len(args) <= 4:
            raise ValueError(f"'{env_var}': pulse() needs 3-4 args (given {args})")
        offset, dur, amp = args[0], args[1], args[2]
        period = args[3] if len(args) > 3 else 365.0
        if amp < 0 or period <= 0 or period <= offset or dur <= 0 or dur > period:
            raise ValueError(f"'{env_var}': invalid pulse() args {args}")

        if offset + dur <= period:
            def _pulse(t, o=offset, d=dur, a=amp, p=period):
                return a if (o <= (t % p) <= o + d) else 0.0
        else:  # wrap-around pulse
            def _pulse(t, o=offset, d=dur, a=amp, p=period):
                tm = t % p
                return a if (o <= tm or tm <= o + d - p) else 0.0
        return _pulse

    if fct_type == "ou_process":
        raise NotImplementedError(f"'{env_var}': OU_process is unimplemented upstream too")
    raise ValueError(f"Unknown function type '{fct_type}' for '{env_var}'")


def make_fluctuation_function(env_var: str, spec) -> FluctFn:
    """Build a time-function from a config spec (number or ``'a + b + ...'`` string).

    Components separated by ``+`` are summed, matching ``mcom_io.jl``.
    """
    if isinstance(spec, (int, float)):
        return _make_component(env_var, str(spec))
    components = [c for c in str(spec).split("+")]
    fns = [_make_component(env_var, c) for c in components]
    return lambda t, _fns=tuple(fns): sum(f(t) for f in _fns)


# --------------------------------------------------------------------------- #
# Parameter container
# --------------------------------------------------------------------------- #


@dataclass
class MComParams:
    """Instantiated MCoM parameter set (all arrays 0-indexed)."""

    # dimensions
    nP: int
    nH: int
    nM: int
    nDOC: int
    nDON: int

    # variant flags
    use_PI_curve: bool
    use_exudation: bool
    surplus_remineralization: bool

    # phytoplankton (vectors length nP)
    I_max: np.ndarray
    I_alpha: np.ndarray
    I_beta: np.ndarray
    V_NP: np.ndarray
    K_NP: np.ndarray
    XP_C: np.ndarray
    XP_N: np.ndarray
    rCN_P: np.ndarray
    rCN_ex: np.ndarray
    q_ex: np.ndarray
    d_P: np.ndarray
    d_P2: np.ndarray
    lambda_: np.ndarray
    rChlC: np.ndarray
    A_MP: np.ndarray   # [nM, nP]
    h_M: np.ndarray    # [nM, nP]
    R_PDOC: np.ndarray  # [nP, nDOC]
    R_PDON: np.ndarray  # [nP, nDON]

    # heterotrophs
    XH_C: np.ndarray
    XH_N: np.ndarray
    rCN_H: np.ndarray
    V_NH: np.ndarray
    K_NH: np.ndarray
    V_DOCH: np.ndarray  # [nDOC, nH]
    V_DONH: np.ndarray  # [nDON, nH]
    K_DOCH: np.ndarray  # [nDOC, nH]
    K_DONH: np.ndarray  # [nDON, nH]
    Y_DOC: np.ndarray   # [nDOC, nH]
    Y_DON: np.ndarray   # [nDON, nH]
    P_HM: np.ndarray    # [nH, nM]
    P_HM_total: np.ndarray
    d_H: np.ndarray
    d_H2: np.ndarray
    R_HDOC: np.ndarray  # [nH, nDOC]
    R_HDON: np.ndarray  # [nH, nDON]

    # environment
    D: float
    d_M: np.ndarray
    SourceNrnd: float
    irradiance_fn: FluctFn
    nutrient_fn: FluctFn
    temperature_fn: FluctFn | None

    # initial state
    t_init: float
    P_init: np.ndarray
    H_init: np.ndarray
    M_init: np.ndarray
    DOC_init: np.ndarray
    DON_init: np.ndarray
    N_init: float


@dataclass
class MComState:
    """Community state vector (mirrors ``State`` in mcom_types.jl)."""

    t: float
    P: np.ndarray
    H: np.ndarray
    M: np.ndarray
    DOC: np.ndarray
    DON: np.ndarray
    N: float
    Cin_total: float = 0.0
    Cout_total: float = 0.0

    def copy(self) -> "MComState":
        return MComState(
            self.t, self.P.copy(), self.H.copy(), self.M.copy(),
            self.DOC.copy(), self.DON.copy(), self.N, self.Cin_total, self.Cout_total,
        )


@dataclass
class Derivs:
    P: np.ndarray
    H: np.ndarray
    M: np.ndarray
    DOC: np.ndarray
    DON: np.ndarray
    N: float
    Cin: float
    Cout: float

    def copy(self) -> "Derivs":
        return Derivs(self.P.copy(), self.H.copy(), self.M.copy(),
                      self.DOC.copy(), self.DON.copy(), self.N, self.Cin, self.Cout)


# --------------------------------------------------------------------------- #
# Right-hand side (reproduces mcom_system.jl:calculate_derivates!)
# --------------------------------------------------------------------------- #


def derivatives(state: MComState, p: MComParams, *, irradiance=None, nutrient=None,
                temperature=None, rng: np.random.Generator | None = None) -> Derivs:
    """Compute d/dt of the full community state.

    Verbatim port of ``calculate_derivates!`` and its callees. Optional
    ``irradiance`` / ``nutrient`` / ``temperature`` floats override the config
    fluctuation functions for this evaluation (the composable input channel);
    ``None`` means "use the config function".
    """
    t = state.t
    irrad = float(p.irradiance_fn(t)) if irradiance is None else float(irradiance)
    # temperature modulation is a stub upstream (tempfac == 1.0); kept identical.
    tempfac = 1.0

    P = np.maximum(state.P, 0.0)
    H = np.maximum(state.H, 0.0)
    M = np.maximum(state.M, 0.0)
    DOC = np.maximum(state.DOC, 0.0)
    DON = np.maximum(state.DON, 0.0)
    N = max(state.N, 0.0)

    # --- accumulators shared across pools (mcom_types.jl:Flows) ---------------
    G_P = np.zeros(p.nP)
    L_P = np.zeros(p.nP)
    f_exDOC_P = np.zeros(p.nP)
    f_exDON_P = np.zeros(p.nP)
    f_upN_P = 0.0
    f_upC_P = 0.0

    G_H = np.zeros(p.nH)
    L_H = np.zeros(p.nH)
    f_upDOC = np.zeros(p.nDOC)
    f_upDON = np.zeros(p.nDON)
    f_upN_H = 0.0
    f_exC_H = 0.0
    f_exN_H = 0.0

    # --- phytoplankton (calculate_phytoplankton_tendencies!) ------------------
    dPdt = np.zeros(p.nP)
    for i in range(p.nP):
        d_P2_cell = p.d_P2[i] * p.XP_C[i]
        mi = float(np.sum(p.A_MP[:, i] * M / (p.h_M[:, i] + M)))
        li = max(0.0, p.d_P[i] + d_P2_cell * P[i] + p.lambda_[i] * mi)

        fN_max_up = tempfac * p.V_NP[i] * N / (p.K_NP[i] + N)
        if p.use_PI_curve:
            I_max_cell = p.I_max[i] * p.rChlC[i] * p.XP_C[i]
            psi = (tempfac * I_max_cell
                   * (1.0 - math.exp(-irrad * p.I_alpha[i] / p.I_max[i]))
                   * math.exp(-p.I_beta[i] * irrad / p.I_max[i]))
        else:
            psi = tempfac * irrad * p.XP_C[i] * p.rChlC[i]

        fC_up = psi
        fDOC_ex = fC_up - p.rCN_P[i] * fN_max_up

        if fDOC_ex >= 0.0:
            # Case 1: N-uptake limits; only DOC exuded.
            fN_up = fN_max_up
            fC_in = p.rCN_P[i] * fN_up
            fDON_ex = 0.0
            fN_in = fN_up
        elif p.use_exudation:
            # Case 2.1: photosynthesis limits; DON exudation coupled to DOC.
            q = p.q_ex[i]
            fN_up_lim = ((p.rCN_ex[i] + q * (p.rCN_P[i] - p.rCN_ex[i])) * fC_up
                         / (p.rCN_P[i] * p.rCN_ex[i]))
            fN_up = min(fN_max_up, fN_up_lim)
            fDON_ex = (p.rCN_P[i] * fN_up - fC_up) / (p.rCN_P[i] - p.rCN_ex[i])
            fDOC_ex = p.rCN_ex[i] * fDON_ex
            fN_in = fN_up - fDON_ex
            fC_in = p.rCN_P[i] * fN_in
        else:
            # Case 2.2: photosynthesis limits; DON / DOC exudation uncoupled.
            fN_up = fN_max_up
            fC_in = fC_up
            fDOC_ex = 0.0
            fN_in = fC_in / p.rCN_P[i]
            fDON_ex = fN_up - fN_in

        gi = fC_in / p.XP_C[i]
        G_P[i] = gi * P[i]
        L_P[i] = li * P[i]
        f_upN_P += fN_up * P[i]
        f_upC_P += fC_up * P[i]
        f_exDOC_P[i] = fDOC_ex * P[i]
        f_exDON_P[i] = fDON_ex * P[i]
        dPdt[i] = G_P[i] - L_P[i] - p.D * P[i]

    # --- heterotrophs (calculate_heterotrophs_tendencies!) --------------------
    dHdt = np.zeros(p.nH)
    for k in range(p.nH):
        fDOC_up_max = tempfac * p.V_DOCH[:, k] * DOC / (p.K_DOCH[:, k] + DOC)
        fDOC_in_max = fDOC_up_max * p.Y_DOC[:, k]
        fDON_up_max = tempfac * p.V_DONH[:, k] * DON / (p.K_DONH[:, k] + DON)
        fDON_in_max = fDON_up_max * p.Y_DON[:, k]
        fDIN_in_max = tempfac * p.V_NH[k] * N / (p.K_NH[k] + N)

        fDOC_in_max_total = float(np.sum(fDOC_in_max))
        fDON_in_max_total = float(np.sum(fDON_in_max))
        fN_in_max_total = fDIN_in_max + fDON_in_max_total
        fDIN_up = fDIN_in_max

        if fDOC_in_max_total < p.rCN_H[k] * fN_in_max_total:
            # Case 1: DOC limits; N-assimilation tied to C-assimilation.
            fC_in = fDOC_in_max_total
            fN_in = fC_in / p.rCN_H[k]
            if fN_in < fDON_in_max_total:
                if p.surplus_remineralization:
                    fDON_up = fDON_up_max
                else:
                    ck = fN_in / fDON_in_max_total
                    fDON_up = ck * fDON_up_max
            else:
                fDON_up = fDON_up_max
            fDOC_up = fDOC_up_max
        else:
            # Case 2: N (DIN + DON) limits; DOC uptake down-regulated.
            fDON_up = fDON_up_max
            fN_in = fN_in_max_total
            fC_in = fN_in_max_total * p.rCN_H[k]
            # When fDOC_in_max_total == 0 the uptake is already zero (DOC ~ 0),
            # so ck -> 0 leaves every flow unchanged while avoiding a 0/0 NaN.
            ck = fC_in / fDOC_in_max_total if fDOC_in_max_total > 0.0 else 0.0
            fDOC_up = ck * fDOC_up_max

        fN_ex = float(np.sum(fDON_up)) + fDIN_up - fN_in
        f_exN_H += fN_ex * H[k]
        fC_ex = float(np.sum(fDOC_up)) - fC_in
        f_exC_H += fC_ex * H[k]

        f_upDOC += fDOC_up * H[k]
        f_upDON += fDON_up * H[k]
        f_upN_H += fDIN_up * H[k]

        gk = fC_in / p.XH_C[k]
        G_H[k] = gk * H[k]
        d_H2_cell = p.d_H2[k] * p.XH_C[k]
        lk = p.d_H[k] + d_H2_cell * H[k] + gk * p.P_HM_total[k]
        L_H[k] = lk * H[k]
        dHdt[k] = G_H[k] - L_H[k] - p.D * H[k]

    # --- metabolites (calculate_metabolite_tendencies!) -----------------------
    dMdt = np.zeros(p.nM)
    for l in range(p.nM):
        prod = float(np.sum(p.P_HM[:, l] * G_H * p.XH_C))
        dMdt[l] = prod - p.d_M[l] * state.M[l] - p.D * state.M[l]

    # --- DOC (calculate_DOC_tendencies!) --------------------------------------
    exudP_DOC = (p.R_PDOC.T @ (f_exDOC_P + L_P * p.XP_C))
    exudH_DOC = (p.R_HDOC.T @ (L_H * p.XH_C))
    dDOCdt = exudP_DOC + exudH_DOC - f_upDOC - p.D * state.DOC

    # --- DON (calculate_DON_tendencies!) --------------------------------------
    exudP_DON = (p.R_PDON.T @ (f_exDON_P + L_P * p.XP_N))
    exudH_DON = (p.R_HDON.T @ (L_H * p.XH_N))
    dDONdt = exudP_DON + exudH_DON - f_upDON - p.D * state.DON

    # --- inorganic nitrogen (calculate_N_tendencies!) -------------------------
    if rng is not None and p.SourceNrnd != 0.0:
        source_err = p.SourceNrnd * (0.5 - rng.random())
    else:
        source_err = 0.0
    N_ext = float(p.nutrient_fn(t)) if nutrient is None else float(nutrient)
    N_ext = max(0.0, N_ext + source_err)
    dNdt = f_exN_H - f_upN_H - f_upN_P + p.D * (N_ext - state.N)

    # --- carbon bookkeeping (calculate_carbon_tendencies) ---------------------
    diffC = p.D * (float(np.sum(state.P * p.XP_C)) + float(np.sum(state.H * p.XH_C))
                   + float(np.sum(state.DOC)))
    dCin = f_upC_P
    dCout = f_exC_H + diffC

    return Derivs(dPdt, dHdt, dMdt, dDOCdt, dDONdt, dNdt, dCin, dCout)


# --------------------------------------------------------------------------- #
# Engine: 2nd-order Adams-Bashforth integrator (reproduces main.jl loop)
# --------------------------------------------------------------------------- #


class MComEngine:
    """Owns the authoritative community state and steps it with AB2.

    The engine keeps the previous derivative (``_ddt_prev``) so that
    ``advance()`` can be called repeatedly (once per bigraph interval) while
    preserving Adams-Bashforth continuity, exactly as ``main.jl`` does across
    its inner loop.
    """

    def __init__(self, params: MComParams, dt: float,
                 seed: int | None = None, source_noise: bool = False):
        self.p = params
        self.dt = float(dt)
        self.state = MComState(
            t=params.t_init,
            P=params.P_init.copy(), H=params.H_init.copy(), M=params.M_init.copy(),
            DOC=params.DOC_init.copy(), DON=params.DON_init.copy(), N=params.N_init,
        )
        self._rng = np.random.default_rng(seed) if (source_noise and params.SourceNrnd) else None
        # Bootstrap: ddt_prev initialised with the initial derivative, so the
        # first AB2 step reduces to forward Euler (matches main.jl).
        self._ddt_prev = derivatives(self.state, self.p, rng=self._rng)

    def _rhs(self, irradiance=None, nutrient=None, temperature=None) -> Derivs:
        return derivatives(self.state, self.p, irradiance=irradiance,
                           nutrient=nutrient, temperature=temperature, rng=self._rng)

    def step(self, dt: float, **forcings) -> None:
        """Advance the state by a single Adams-Bashforth step of width ``dt``."""
        ddt = self._rhs(**forcings)
        s, prev = self.state, self._ddt_prev

        def ab(x, d, dp):
            return x + (1.5 * d - 0.5 * dp) * dt

        s.t += dt
        s.N = ab(s.N, ddt.N, prev.N)
        s.Cin_total = ab(s.Cin_total, ddt.Cin, prev.Cin)
        s.Cout_total = ab(s.Cout_total, ddt.Cout, prev.Cout)
        s.P = ab(s.P, ddt.P, prev.P)
        s.H = ab(s.H, ddt.H, prev.H)
        s.M = ab(s.M, ddt.M, prev.M)
        s.DOC = ab(s.DOC, ddt.DOC, prev.DOC)
        s.DON = ab(s.DON, ddt.DON, prev.DON)
        self._ddt_prev = ddt.copy()

    def advance(self, duration: float, **forcings) -> None:
        """Advance the state by ``duration`` days using internal ``dt`` substeps.

        ``duration`` is split into an integer number of substeps that land
        exactly on ``duration`` (so a bigraph interval equal to the config's
        ``dt_out`` reproduces the upstream save grid).
        """
        if duration <= 0:
            return
        nsub = max(1, int(round(duration / self.dt)))
        sub_dt = duration / nsub
        for _ in range(nsub):
            self.step(sub_dt, **forcings)

    def run(self, t_end: float, dt_out: float, save_offset: float = 0.0):
        """Run standalone from the initial state to ``t_end``.

        Returns a dict of trajectories keyed ``t, P, H, M, DOC, DON, N, Cin, Cout``,
        saved every ``dt_out`` days once ``t >= save_offset`` -- matching
        ``main.jl``'s save cadence. Used for validation and standalone plots.
        """
        traj = {k: [] for k in ("t", "P", "H", "M", "DOC", "DON", "N", "Cin", "Cout")}

        def record():
            s = self.state
            traj["t"].append(s.t)
            traj["P"].append(s.P.copy())
            traj["H"].append(s.H.copy())
            traj["M"].append(s.M.copy())
            traj["DOC"].append(s.DOC.copy())
            traj["DON"].append(s.DON.copy())
            traj["N"].append(s.N)
            traj["Cin"].append(s.Cin_total)
            traj["Cout"].append(s.Cout_total)

        nsteps = int(math.ceil(round((t_end - self.state.t) / self.dt, 1)))
        if self.state.t >= save_offset:
            record()
            next_save = self.state.t + dt_out - self.dt * 0.5
        else:
            next_save = save_offset - self.dt * 0.5

        for _ in range(nsteps):
            self.step(self.dt)
            if self.state.t >= next_save:
                record()
                next_save += dt_out

        for k, v in traj.items():
            traj[k] = np.array(v)
        return traj
