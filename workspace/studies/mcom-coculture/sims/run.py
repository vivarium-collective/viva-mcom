#!/usr/bin/env python3
"""Canonical run for the ``mcom-coculture`` study.

Builds the ``mcom_coculture`` composite (a single ``MComReproductionProcess``
carrying MCoM's shipped ``configs/test.toml`` 1-phytoplankton / 1-heterotroph
batch community -- a clean-room NumPy reproduction of the Julia engine), runs it
for 200 days, and:

1. Derives the two behaviour-test scalars (run-mean total heterotroph density;
   second-half-mean inorganic nitrogen).
2. Renders an interactive succession figure (phytoplankton + heterotroph vs
   inorganic nitrogen) and a DOC/DON pool figure into ``viz/``.
3. Records the run into ``.pbg/runs.jsonl``.
4. Prints PASS/FAIL mirroring the study's declared behavior_tests exactly.

Standalone; run from the workspace root::

    python workspace/studies/mcom-coculture/sims/run.py
"""
from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import numpy as np
from process_bigraph import Composite, gather_emitter_results

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]

from viva_mcom.core import build_core
from viva_mcom.composites.coculture import mcom_coculture
from viva_mcom import viz
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_mcom.composites.coculture.mcom_coculture"
STUDY_SLUG = "mcom-coculture"
INVESTIGATION_SLUG = "mcom-showcase"

T_END = 200.0
# Behaviour-test thresholds (see study.yaml behavior_tests for rationale).
H_MEAN_THRESHOLD = 0.1   # heterotroph-blooms-on-exudate: mean(H_total) > 0.1
DIN_MEAN_THRESHOLD = 4.0  # inorganic-nitrogen-drawn-down: mean(DIN, 2nd half) < 4.0


def main() -> int:
    run_id = uuid.uuid4().hex
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID,
        "label": STUDY_SLUG, "started_at": time.time(), "status": "running",
        "n_steps": int(T_END / 0.1), "emitter": "ram", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": {"t_end": T_END},
    })
    try:
        core = build_core()
        sim = Composite({"state": mcom_coculture(core, t_end=T_END)}, core=core)
        sim.run(T_END)
        rows = gather_emitter_results(sim)[("emitter",)]
        if len(rows) < 10:
            raise RuntimeError(f"only {len(rows)} rows emitted")

        t = np.array([r["global_time"] for r in rows])
        P = np.array([r["P_total"] for r in rows])
        H = np.array([r["H_total"] for r in rows])
        DIN = np.array([r["DIN"] for r in rows])
        DOC0 = np.array([r["DOC"]["DOC0"] for r in rows])
        DOC1 = np.array([r["DOC"]["DOC1"] for r in rows])
        DON0 = np.array([r["DON"]["DON0"] for r in rows])

        half = t >= t[-1] / 2
        h_mean = float(H.mean())
        din_half_mean = float(DIN[half].mean())
        p_peak, p_peak_t = float(P.max()), float(t[int(P.argmax())])
        h_peak, h_peak_t = float(H.max()), float(t[int(H.argmax())])

        heterotroph_grows = h_mean > H_MEAN_THRESHOLD
        nitrogen_drawn_down = din_half_mean < DIN_MEAN_THRESHOLD

        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        (viz_dir / "succession.html").write_text(viz.succession_html(
            t, P, H, DIN,
            "MCoM batch co-culture: phototroph-heterotroph succession",
            subtitle=(f"phytoplankton peak {p_peak:.2f}@day{p_peak_t:.0f}, "
                      f"heterotroph peak {h_peak:.2f}@day{h_peak_t:.0f} (lags the bloom); "
                      "DIN drawn down (reproduction of MCoM configs/test.toml)"),
            log_pops=True))
        (viz_dir / "organic_pools.html").write_text(viz.pools_html(
            t, {"DOC0 (labile)": DOC0, "DOC1 (recalcitrant)": DOC1},
            {"DON0": DON0},
            "Dissolved organic pools: labile consumed, recalcitrant accumulates",
            subtitle="heterotroph consumes DOC0/DON0; its mortality routes 1% of C to DOC1"))
    except Exception:
        append_run_event(WORKSPACE_ROOT, {
            "run_id": run_id, "event": "completed", "completed_at": time.time(),
            "n_steps": 0, "status": "failed"})
        raise

    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "completed", "completed_at": time.time(),
        "n_steps": len(rows), "status": "completed"})

    print(f"[mcom-coculture] phytoplankton peak {p_peak:.3f} @ day {p_peak_t:.0f}; "
          f"heterotroph peak {h_peak:.3f} @ day {h_peak_t:.0f} (lag {h_peak_t - p_peak_t:.0f} d)")
    print(f"[mcom-coculture] mean(H_total) = {h_mean:.3f} -> "
          f"{'PASS' if heterotroph_grows else 'FAIL'} (heterotroph-blooms-on-exudate > {H_MEAN_THRESHOLD})")
    print(f"[mcom-coculture] mean(DIN, 2nd half) = {din_half_mean:.3f} -> "
          f"{'PASS' if nitrogen_drawn_down else 'FAIL'} (inorganic-nitrogen-drawn-down < {DIN_MEAN_THRESHOLD})")
    print(f"[mcom-coculture] viz: {viz_dir / 'succession.html'}, {viz_dir / 'organic_pools.html'}")
    print(f"[mcom-coculture] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")
    return 0 if (heterotroph_grows and nitrogen_drawn_down) else 1


if __name__ == "__main__":
    raise SystemExit(main())
