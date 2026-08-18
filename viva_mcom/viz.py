"""Interactive Plotly visualizations for MCoM community runs.

Self-contained figure builders returning HTML strings (Plotly loaded from the
pinned v3 CDN that matches plotly.py 6.x's numpy typed-array encoding). Kept
deliberately small and considered: one succession figure and one nutrient/pool
figure, with a colourblind-safe palette and a clean, legible layout.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# Colourblind-safe categorical palette (phototroph greens, heterotroph amber,
# nutrient blue, organic-pool purples).
COL = {
    "phyto": "#2E8B6B",
    "hetero": "#D08A2E",
    "din": "#3A6EA5",
    "doc": "#8E5FA8",
    "don": "#4CA6A0",
    "grid": "rgba(130,130,130,0.18)",
    "ink": "#2b2b33",
}

_FONT = dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif",
             size=13, color=COL["ink"])


def _base_layout(title, subtitle=None):
    ttl = f"<b>{title}</b>"
    if subtitle:
        ttl += f"<br><span style='font-size:12px;color:#7a7a86'>{subtitle}</span>"
    return dict(
        title=dict(text=ttl, x=0.02, xanchor="left", y=0.95),
        font=_FONT,
        paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=70, r=70, t=80, b=55),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1.0,
                    bgcolor="rgba(255,255,255,0)"),
    )


def _axis(title, **kw):
    base = dict(title=title, showgrid=True, gridcolor=COL["grid"],
                zeroline=False, linecolor="rgba(130,130,130,0.4)",
                ticks="outside", tickcolor="rgba(130,130,130,0.4)")
    base.update(kw)  # caller overrides (e.g. showgrid=False on the overlay axis)
    return base


def succession_html(t, P, H, DIN, title, subtitle=None, log_pops=True):
    """Phytoplankton + heterotroph (left axis, optionally log) vs inorganic
    nitrogen (right axis) -- the phototroph->heterotroph succession figure."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=P, name="Phytoplankton", mode="lines",
                             line=dict(color=COL["phyto"], width=2.4),
                             hovertemplate="P=%{y:.4g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=H, name="Heterotroph", mode="lines",
                             line=dict(color=COL["hetero"], width=2.4),
                             hovertemplate="H=%{y:.4g}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=DIN, name="Inorganic N (DIN)", mode="lines",
                             yaxis="y2", line=dict(color=COL["din"], width=1.8, dash="dot"),
                             hovertemplate="DIN=%{y:.4g}<extra></extra>"))
    layout = _base_layout(title, subtitle)
    layout["xaxis"] = _axis("time [days]")
    layout["yaxis"] = _axis("cell density [1e6 cells/mL]" + (" (log)" if log_pops else ""),
                            type="log" if log_pops else "linear")
    layout["yaxis2"] = _axis("DIN [mM N]", overlaying="y", side="right", showgrid=False)
    fig.update_layout(**layout)
    return fig.to_html(include_plotlyjs="cdn", full_html=True,
                       config={"displaylogo": False, "responsive": True})


def pools_html(t, doc_series, don_series, title, subtitle=None):
    """Dissolved organic carbon and nitrogen pools over time (each a dict
    ``{label: array}``)."""
    fig = go.Figure()
    for i, (label, arr) in enumerate(doc_series.items()):
        fig.add_trace(go.Scatter(x=t, y=arr, name=label, mode="lines",
                                 line=dict(color=COL["doc"], width=2.0,
                                           dash="solid" if i == 0 else "dash")))
    for i, (label, arr) in enumerate(don_series.items()):
        fig.add_trace(go.Scatter(x=t, y=arr, name=label, mode="lines",
                                 line=dict(color=COL["don"], width=2.0,
                                           dash="solid" if i == 0 else "dash")))
    layout = _base_layout(title, subtitle)
    layout["xaxis"] = _axis("time [days]")
    layout["yaxis"] = _axis("concentration [mM]")
    fig.update_layout(**layout)
    return fig.to_html(include_plotlyjs="cdn", full_html=True,
                       config={"displaylogo": False, "responsive": True})
