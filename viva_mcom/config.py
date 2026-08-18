"""Instantiate an :class:`~viva_mcom.engine.MComParams` from an MCoM TOML config.

Clean-room reproduction of the *deterministic* subset of MCoM's parameter
instantiation (``mcom_generator.jl`` + ``mcom_io.jl``). Supported value forms
per parameter:

* **scalar** -- broadcast/uniform fill;
* **``[min, max]`` range** -- reproduced only when ``min == max`` (a degenerate,
  deterministic draw); a genuine ``min < max`` range needs Julia's RNG and raises
  :class:`NotImplementedError`;
* **explicit** (name listed in a section's ``explicit = [...]``) -- literal values
  (a vector for 1D params, a list-of-lists mapped onto the connectivity for 2D);
* **competing_pairs** -- the interaction matrices become diagonal.

The random Dirichlet/simplex splitting of un-normalised release/rate matrices
across more than one connected compound is *not* reproduced (documented
divergence); single-edge and explicit networks are exact.
"""

from __future__ import annotations

import numpy as np

from .engine import MComParams, make_fluctuation_function

# name -> (dim0_key, dim1_key, kind, connectivity_key)
_MATRIX_META = {
    "R_PDOC": ("P", "DOC", "release", "exPDOC"),
    "R_PDON": ("P", "DON", "release", "exPDON"),
    "R_HDOC": ("H", "DOC", "release", "exHDOC"),
    "R_HDON": ("H", "DON", "release", "exHDON"),
    "V_DOCH": ("DOC", "H", "rate", "upDOCH"),
    "V_DONH": ("DON", "H", "rate", "upDONH"),
    "P_HM":   ("H", "M", "rate", "prodHM"),
    "K_DOCH": ("DOC", "H", "other", "upDOCH"),
    "K_DONH": ("DON", "H", "other", "upDONH"),
    "Y_DOC":  ("DOC", "H", "other", "upDOCH"),
    "Y_DON":  ("DON", "H", "other", "upDONH"),
    "A_MP":   ("M", "P", "other", "effsMP"),
    "h_M":    ("M", "P", "other", "effsMP"),
}

_COMPETING_PAIRS_DIAG = set(_MATRIX_META)  # all interaction matrices go diagonal

# connectivity key -> (population_dim_key, compound_dim_key)
_CONN_META = {
    "exPDOC": ("P", "DOC"), "exPDON": ("P", "DON"),
    "exHDOC": ("H", "DOC"), "exHDON": ("H", "DON"),
    "upDOCH": ("H", "DOC"), "upDONH": ("H", "DON"),
    "prodHM": ("H", "M"), "effsMP": ("P", "M"),
}


def _resolve_scalar(name, value):
    """Return a single float from a scalar or a degenerate ``[v, v]`` range."""
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, (list, tuple)):
        if len(value) == 2 and float(value[0]) == float(value[1]):
            return float(value[0])
        raise NotImplementedError(
            f"Parameter '{name}': a non-degenerate random range {list(value)} requires "
            "Julia's RNG and is not reproduced. Provide a scalar, a degenerate range "
            "[v, v], or explicit values."
        )
    raise ValueError(f"Parameter '{name}': unsupported value {value!r}")


def _instantiate_vector(name, value, dim, explicit_set):
    if name in explicit_set:
        arr = np.asarray(value, dtype=float).ravel()
        if arr.size != dim:
            raise ValueError(f"Explicit vector '{name}' must have length {dim}, got {arr.size}")
        return arr
    return np.full(dim, _resolve_scalar(name, value), dtype=float)


def _instantiate_matrix(name, value, dims, connectivity, explicit_set, competing_pairs):
    d0k, d1k, kind, conn_key = _MATRIX_META[name]
    shape = (dims[d0k], dims[d1k])
    pop_axis = 0 if d0k in ("P", "H") else 1
    n_pop = shape[pop_axis]
    conn = connectivity[conn_key]  # list (len n_pop) of 0-based compound-index lists

    M = np.zeros(shape, dtype=float)

    if name in explicit_set:
        vals = value
        if len(vals) != n_pop:
            raise ValueError(f"Explicit '{name}' must be a list of {n_pop} vectors")
        for i in range(n_pop):
            idx = conn[i]
            row = np.asarray(vals[i], dtype=float).ravel()
            if row.size != len(idx):
                raise ValueError(
                    f"Explicit '{name}'[{i}] has {row.size} values but {len(idx)} links")
            if pop_axis == 0:
                M[i, idx] = row
            else:
                M[idx, i] = row
        return M

    if competing_pairs and name in _COMPETING_PAIRS_DIAG:
        if shape[0] != shape[1]:
            raise ValueError(f"competing_pairs requires square '{name}', got {shape}")
        v = _resolve_scalar(name, value)
        np.fill_diagonal(M, v)
        return M

    if kind == "other":
        v = _resolve_scalar(name, value)
        for i in range(n_pop):  # population is dim1 for every 'other' matrix
            for j in conn[i]:
                M[j, i] = v
        return M

    if kind == "release":
        equi = _resolve_scalar(name, value)
        for i in range(n_pop):
            idx = conn[i]
            m = len(idx)
            if m == 0:
                continue
            if m == 1:
                frac = np.array([1.0])            # single edge -> deterministic 1.0
            elif equi == 1.0:
                frac = np.full(m, 1.0 / m)         # fully-even split -> deterministic
            else:
                raise NotImplementedError(
                    f"Release matrix '{name}': Dirichlet split across {m} compounds "
                    f"(equi_fraction={equi}) needs Julia's RNG. Provide explicit fractions.")
            if pop_axis == 0:
                M[i, idx] = frac
            else:
                M[idx, i] = frac
        return M

    # kind == "rate"
    total = _resolve_scalar(name, value)
    for i in range(n_pop):
        idx = conn[i]
        m = len(idx)
        if m == 0:
            continue
        if m == 1:
            split = np.array([1.0])                # single edge -> total on that edge
        else:
            raise NotImplementedError(
                f"Rate matrix '{name}': simplex split across {m} compounds needs Julia's "
                f"RNG. Provide explicit per-edge rates.")
        if pop_axis == 0:
            M[i, idx] = total * split
        else:
            M[idx, i] = total * split
    return M


def _full_connectivity(dims):
    conn = {}
    for key, (_pk, ck) in _CONN_META.items():
        n_pop = dims[_CONN_META[key][0]]
        conn[key] = [list(range(dims[ck])) for _ in range(n_pop)]
    return conn


def _instantiate_connectivity(network, dims):
    if network is None:
        return _full_connectivity(dims)
    if not network.get("explicit", False):
        # numeric link-counts -> would need RNG to place; only full == dims is safe.
        conn = {}
        for key, (pk, ck) in _CONN_META.items():
            n_pop = dims[pk]
            val = network.get(key)
            if val is None:
                raise ValueError(f"[network] missing '{key}'")
            if isinstance(val, list):
                raise NotImplementedError(
                    "Randomised link counts in [network] need Julia's RNG; use explicit=true")
            if int(val) != dims[ck]:
                raise NotImplementedError(
                    f"[network] '{key}={val}' selects a random subset of {dims[ck]} compounds "
                    "(needs RNG). Use explicit=true or connect all compounds.")
            conn[key] = [list(range(dims[ck])) for _ in range(n_pop)]
        return conn
    # explicit: 1-based index lists -> 0-based
    conn = {}
    for key, (pk, ck) in _CONN_META.items():
        raw = network[key]
        conn[key] = [[int(j) - 1 for j in lst] for lst in raw]
    return conn


def instantiate_params(config: dict) -> MComParams:
    """Build an :class:`MComParams` from a parsed MCoM config dict."""
    dims = {k: int(config["dims"][k]) for k in ("P", "H", "M", "DOC", "DON")}
    variant = config.get("variant", {})
    competing_pairs = bool(variant.get("competing_pairs", False))
    use_PI_curve = bool(variant.get("use_PI_curve", False))
    use_exudation = bool(variant.get("use_exudation", False))
    surplus_remin = bool(variant.get("surplus_remineralization", True))

    connectivity = _instantiate_connectivity(config.get("network"), dims)

    dyn = config["dynamics"]
    dyn_explicit = set(dyn.get("explicit", []))
    init = config["init"]
    init_explicit = set(init.get("explicit", []))
    env = config["environment"]
    env_explicit = set(env.get("explicit", []))

    def vec(section, name, dim, explicit_set):
        return _instantiate_vector(name, section[name], dim, explicit_set)

    def mat(name):
        return _instantiate_matrix(name, dyn[name], dims, connectivity,
                                   dyn_explicit, competing_pairs)

    # phytoplankton vectors
    XP_C = vec(dyn, "XP_C", dims["P"], dyn_explicit)
    rCN_P = vec(dyn, "rCN_P", dims["P"], dyn_explicit)

    # heterotroph vectors
    XH_C = vec(dyn, "XH_C", dims["H"], dyn_explicit)
    rCN_H = vec(dyn, "rCN_H", dims["H"], dyn_explicit)

    P_HM = mat("P_HM")

    # environment vectors + fluctuations
    d_M = _instantiate_vector("d_M", env["d_M"], dims["M"], env_explicit)

    K_DOCH = mat("K_DOCH")
    K_DONH = mat("K_DONH")
    h_M = mat("h_M")
    # impute_nonzero_Ks!: K == 0 -> 1e12 (avoid div-by-zero at vanishing conc.)
    for K in (K_DOCH, K_DONH, h_M):
        K[K == 0.0] = 1e12

    temperature_fn = None
    if "temperature" in env:
        temperature_fn = make_fluctuation_function("temperature", env["temperature"])

    return MComParams(
        nP=dims["P"], nH=dims["H"], nM=dims["M"], nDOC=dims["DOC"], nDON=dims["DON"],
        use_PI_curve=use_PI_curve, use_exudation=use_exudation,
        surplus_remineralization=surplus_remin,
        # phytoplankton
        I_max=vec(dyn, "I_max", dims["P"], dyn_explicit),
        I_alpha=vec(dyn, "I_alpha", dims["P"], dyn_explicit),
        I_beta=vec(dyn, "I_beta", dims["P"], dyn_explicit),
        V_NP=vec(dyn, "V_NP", dims["P"], dyn_explicit),
        K_NP=vec(dyn, "K_NP", dims["P"], dyn_explicit),
        XP_C=XP_C, XP_N=XP_C / rCN_P, rCN_P=rCN_P,
        rCN_ex=vec(dyn, "rCN_ex", dims["P"], dyn_explicit),
        q_ex=vec(dyn, "q_ex", dims["P"], dyn_explicit),
        d_P=vec(dyn, "d_P", dims["P"], dyn_explicit),
        d_P2=vec(dyn, "d_P2", dims["P"], dyn_explicit),
        lambda_=vec(dyn, "lambda", dims["P"], dyn_explicit),
        rChlC=vec(dyn, "rChlC", dims["P"], dyn_explicit),
        A_MP=mat("A_MP"), h_M=h_M,
        R_PDOC=mat("R_PDOC"), R_PDON=mat("R_PDON"),
        # heterotrophs
        XH_C=XH_C, XH_N=XH_C / rCN_H, rCN_H=rCN_H,
        V_NH=vec(dyn, "V_NH", dims["H"], dyn_explicit),
        K_NH=vec(dyn, "K_NH", dims["H"], dyn_explicit),
        V_DOCH=mat("V_DOCH"), V_DONH=mat("V_DONH"),
        K_DOCH=K_DOCH, K_DONH=K_DONH,
        Y_DOC=mat("Y_DOC"), Y_DON=mat("Y_DON"),
        P_HM=P_HM, P_HM_total=P_HM.sum(axis=1),
        d_H=vec(dyn, "d_H", dims["H"], dyn_explicit),
        d_H2=vec(dyn, "d_H2", dims["H"], dyn_explicit),
        R_HDOC=mat("R_HDOC"), R_HDON=mat("R_HDON"),
        # environment
        D=float(env["D"]), d_M=d_M, SourceNrnd=float(env.get("SourceNrnd", 0.0)),
        irradiance_fn=make_fluctuation_function("irradiance", env["irradiance"]),
        nutrient_fn=make_fluctuation_function("nutrient", env["nutrient"]),
        temperature_fn=temperature_fn,
        # initial state
        t_init=float(init.get("t_init", 0.0)),
        P_init=_instantiate_vector("P_init", init["P_init"], dims["P"], init_explicit),
        H_init=_instantiate_vector("H_init", init["H_init"], dims["H"], init_explicit),
        M_init=_instantiate_vector("M_init", init["M_init"], dims["M"], init_explicit),
        DOC_init=_instantiate_vector("DOC_init", init["DOC_init"], dims["DOC"], init_explicit),
        DON_init=_instantiate_vector("DON_init", init["DON_init"], dims["DON"], init_explicit),
        N_init=float(init["N_init"]),
    )
