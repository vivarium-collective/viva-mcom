#!/usr/bin/env python3
"""Canonical run for the ``mcom-seasonal`` study.

Builds the ``mcom_seasonal`` composite (the co-culture parameters in a D = 0.1
chemostat with steady 10 mM nitrogen inflow and seasonally varying irradiance),
runs it for 1460 days (4 seasonal cycles), and:

1. Derives the two behaviour-test scalars over the established (second-half)
   regime (minimum total phytoplankton density; mean inorganic nitrogen) and
   quantifies the seasonal peak/trough swing.
2. Renders an interactive succession figure showing the annual cycle.
3. Records the run into ``.pbg/runs.jsonl``.
4. Prints PASS/FAIL mirroring the study's declared behavior_tests exactly.

Standalone; run from the workspace root::

    python workspace/studies/mcom-seasonal/sims/run.py
"""
from __future__ import annotations

import time
import uuid
from pathlib import Path

import numpy as np
from process_bigraph import Composite, gather_emitter_results

STUDY_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = STUDY_DIR.parents[2]

from viva_mcom.core import build_core
from viva_mcom.composites.coculture import mcom_seasonal
from viva_mcom import viz
from vivarium_workbench.lib.run_log import append_run_event

SPEC_ID = "viva_mcom.composites.coculture.mcom_seasonal"
STUDY_SLUG = "mcom-seasonal"
INVESTIGATION_SLUG = "mcom-showcase"

T_END = 1460.0
N_EXT = 10.0
P_MIN_THRESHOLD = 0.15   # community-sustained-under-dilution: min(P_total, 2nd half) > 0.15
DIN_MEAN_THRESHOLD = 5.0  # nutrient-drawn-down-in-chemostat: mean(DIN, 2nd half) < 5.0


def main() -> int:
    run_id = uuid.uuid4().hex
    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "started", "spec_id": SPEC_ID,
        "label": STUDY_SLUG, "started_at": time.time(), "status": "running",
        "n_steps": int(T_END / 0.1), "emitter": "ram", "origin": "canonical_run",
        "study_slug": STUDY_SLUG, "investigation_slug": INVESTIGATION_SLUG,
        "params": {"t_end": T_END, "nutrient_supply": N_EXT},
    })
    try:
        core = build_core()
        sim = Composite({"state": mcom_seasonal(core, t_end=T_END)}, core=core)
        sim.run(T_END)
        rows = gather_emitter_results(sim)[("emitter",)]
        if len(rows) < 100:
            raise RuntimeError(f"only {len(rows)} rows emitted")

        t = np.array([r["global_time"] for r in rows])
        P = np.array([r["P_total"] for r in rows])
        H = np.array([r["H_total"] for r in rows])
        DIN = np.array([r["DIN"] for r in rows])

        half = t >= t[-1] / 2
        p_min_half = float(P[half].min())
        p_max_half = float(P[half].max())
        din_half_mean = float(DIN[half].mean())
        swing = p_max_half / p_min_half if p_min_half > 0 else float("inf")

        sustained = p_min_half > P_MIN_THRESHOLD
        nutrient_drawn_down = din_half_mean < DIN_MEAN_THRESHOLD

        viz_dir = STUDY_DIR / "viz"
        viz_dir.mkdir(parents=True, exist_ok=True)
        (viz_dir / "seasonal_succession.html").write_text(viz.succession_html(
            t, P, H, DIN,
            "MCoM in a seasonally-lit chemostat: sustained annual succession",
            subtitle=(f"phytoplankton swings {p_min_half:.2f}-{p_max_half:.2f} "
                      f"(peak/trough {swing:.1f}x) each year; DIN held ~{din_half_mean:.1f} mM "
                      f"of a {N_EXT:.0f} mM inflow"),
            log_pops=False))
    except Exception:
        append_run_event(WORKSPACE_ROOT, {
            "run_id": run_id, "event": "completed", "completed_at": time.time(),
            "n_steps": 0, "status": "failed"})
        raise

    append_run_event(WORKSPACE_ROOT, {
        "run_id": run_id, "event": "completed", "completed_at": time.time(),
        "n_steps": len(rows), "status": "completed"})

    print(f"[mcom-seasonal] established regime: P swings {p_min_half:.3f}-{p_max_half:.3f} "
          f"(peak/trough {swing:.2f}x); DIN mean {din_half_mean:.3f} mM (inflow {N_EXT:.0f})")
    print(f"[mcom-seasonal] min(P_total, 2nd half) = {p_min_half:.3f} -> "
          f"{'PASS' if sustained else 'FAIL'} (community-sustained-under-dilution > {P_MIN_THRESHOLD})")
    print(f"[mcom-seasonal] mean(DIN, 2nd half) = {din_half_mean:.3f} -> "
          f"{'PASS' if nutrient_drawn_down else 'FAIL'} (nutrient-drawn-down-in-chemostat < {DIN_MEAN_THRESHOLD})")
    print(f"[mcom-seasonal] viz: {viz_dir / 'seasonal_succession.html'}")
    print(f"[mcom-seasonal] run recorded in {WORKSPACE_ROOT / '.pbg' / 'runs.jsonl'}")
    return 0 if (sustained and nutrient_drawn_down) else 1


if __name__ == "__main__":
    raise SystemExit(main())
