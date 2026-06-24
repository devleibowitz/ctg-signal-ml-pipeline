"""Visualization utilities: interactive before/after signal plots via Plotly."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

_C = {
    "hr1_raw":        "#4682B4",  # steelblue
    "hr1_processed":  "#FF8C00",  # dark orange
    "toco_raw":       "#2E8B57",  # sea green
    "toco_processed": "#E8562A",  # tomato
}

_LINE = dict(width=1)


def _monitor_time_minutes(
    df: pd.DataFrame,
    reference: pd.Timestamp,
) -> pd.Series:
    """Convert Monitor_Date timestamps to elapsed minutes from monitor start."""
    return (pd.to_datetime(df["Monitor_Date"]) - reference).dt.total_seconds() / 60.0


def plot_patient_signals(
    patient_id: str,
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    output_dir: Path | None = None,
    show: bool = False,
) -> go.Figure:
    """Generate an interactive 2-panel before/after comparison for one patient.

    Layout:
        Left  — HR1  raw (blue) + processed (orange) overlaid
        Right — TOCO raw (green) + processed (red-orange) overlaid

    A rangeslider beneath the left panel controls both panels simultaneously.
    Plotly's built-in toolbar (top-right of the figure) provides box-zoom,
    pan, and reset.

    Args:
        patient_id: Used in the figure title and saved filename.
        raw_df: Pre-processing DataFrame with Monitor_Date, HR1, TOCO.
        processed_df: Post-processing DataFrame.
        output_dir: Directory to write an interactive HTML file.  Skipped if None.
        show: Call fig.show() after building (opens browser or renders inline).

    Returns:
        Plotly Figure.
    """
    short_id = patient_id[:30] + "…" if len(patient_id) > 30 else patient_id

    monitor_start = min(
        raw_df["Monitor_Date"].min(),
        processed_df["Monitor_Date"].min(),
    )
    raw_time = _monitor_time_minutes(raw_df, monitor_start)
    processed_time = _monitor_time_minutes(processed_df, monitor_start)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            "HR1 — Fetal Heart Rate (bpm)",
            "TOCO — Tocometry",
        ],
        horizontal_spacing=0.08,
    )

    # ── HR1 panel ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=raw_time, y=raw_df["HR1"],
        name="raw", mode="lines",
        line=dict(**_LINE, color=_C["hr1_raw"]),
        opacity=0.85,
        legendgroup="raw", legendgrouptitle_text="",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=processed_time, y=processed_df["HR1"],
        name="processed", mode="lines",
        line=dict(**_LINE, color=_C["hr1_processed"]),
        opacity=0.9,
        legendgroup="processed",
    ), row=1, col=1)

    # ── TOCO panel ────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=raw_time, y=raw_df["TOCO"],
        name="raw", mode="lines",
        line=dict(**_LINE, color=_C["toco_raw"]),
        opacity=0.85,
        legendgroup="raw",
        showlegend=False,
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=processed_time, y=processed_df["TOCO"],
        name="processed", mode="lines",
        line=dict(**_LINE, color=_C["toco_processed"]),
        opacity=0.9,
        legendgroup="processed",
        showlegend=False,
    ), row=1, col=2)

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text=f"<b>Patient</b> {short_id}  —  raw vs processed",
            font=dict(size=13),
            x=0.5, xanchor="center",
        ),
        height=500,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.04,
            xanchor="right",  x=1,
            font=dict(size=11),
        ),
        # Rangeslider on xaxis drives both panels via matches="x"
        xaxis=dict(
            title="Monitor time (min)",
            rangeslider=dict(visible=True, thickness=0.06),
        ),
        xaxis2=dict(
            title="Monitor time (min)",
            matches="x",
        ),
        yaxis=dict(title="bpm"),
        yaxis2=dict(title="tocometry units"),
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / f"{patient_id}_signals.html"
        fig.write_html(str(html_path))
        logger.info("Saved interactive plot → %s", html_path)

    if show:
        fig.show()

    return fig


def plot_patient_signals_post(
    patient_id: str,
    birth_time: pd.Timestamp,
    raw_df: pd.DataFrame,
    processed_df: pd.DataFrame,
    output_dir: Path | None = None,
    show: bool = False,
) -> go.Figure:
    """2-panel plot with HR1 and TOCO overlaid on the same axis, raw left / processed right.

    HR1 (bpm) is plotted against the left y-axis; TOCO against a secondary right y-axis.
    A vertical dashed line marks the birth time on both panels.
    X-axis is elapsed monitor time in minutes from the earliest recorded sample.

    Args:
        patient_id: Used in the figure title and saved filename.
        birth_time: Timestamp of delivery — drawn as a vertical annotation on both panels.
        raw_df: Pre-processing DataFrame with Monitor_Date, HR1, TOCO.
        processed_df: Post-processing DataFrame.
        output_dir: Directory to write an interactive HTML file.  Skipped if None.
        show: Call fig.show() after building.

    Returns:
        Plotly Figure.
    """
    short_id = patient_id[:30] + "…" if len(patient_id) > 30 else patient_id

    monitor_start = min(
        raw_df["Monitor_Date"].min(),
        processed_df["Monitor_Date"].min(),
    )
    raw_time       = _monitor_time_minutes(raw_df, monitor_start)
    processed_time = _monitor_time_minutes(processed_df, monitor_start)
    birth_min      = (birth_time - monitor_start).total_seconds() / 60.0

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Raw signal", "Processed signal"],
        specs=[[{"secondary_y": True}, {"secondary_y": True}]],
        horizontal_spacing=0.12,
    )

    for col_idx, (x_time, df) in enumerate(
        [(raw_time, raw_df), (processed_time, processed_df)], start=1
    ):
        show_legend = col_idx == 1  # only label in first panel to avoid duplication

        # HR1 — left y-axis
        fig.add_trace(
            go.Scatter(
                x=x_time, y=df["HR1"],
                name="FHR (bpm)", mode="lines",
                line=dict(width=1, color=_C["hr1_raw"] if col_idx == 1 else _C["hr1_processed"]),
                opacity=0.9,
                legendgroup="fhr", showlegend=show_legend,
            ),
            row=1, col=col_idx, secondary_y=False,
        )

        # TOCO — right y-axis
        fig.add_trace(
            go.Scatter(
                x=x_time, y=df["TOCO"],
                name="TOCO", mode="lines",
                line=dict(width=1, color=_C["toco_raw"] if col_idx == 1 else _C["toco_processed"]),
                opacity=0.75,
                legendgroup="toco", showlegend=show_legend,
            ),
            row=1, col=col_idx, secondary_y=True,
        )

        # Birth time vertical line
        fig.add_vline(
            x=birth_min,
            line_dash="dash", line_color="crimson", line_width=1.5,
            annotation_text="birth" if col_idx == 1 else "",
            annotation_position="top left",
            annotation_font=dict(color="crimson", size=10),
            row=1, col=col_idx,
        )

    # Y-axis labels
    fig.update_yaxes(title_text="FHR (bpm)",       secondary_y=False)
    fig.update_yaxes(title_text="TOCO (units)",     secondary_y=True, showgrid=False)
    fig.update_xaxes(title_text="Monitor time (min)", rangeslider=dict(visible=False))
    fig.update_xaxes(matches="x", row=1, col=2)

    fig.update_layout(
        title=dict(
            text=f"<b>Patient</b> {short_id}  —  FHR & TOCO  |  birth at {birth_min:.1f} min",
            font=dict(size=13),
            x=0.5, xanchor="center",
        ),
        height=480,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.04,
            xanchor="right",  x=1,
            font=dict(size=11),
        ),
        # Rangeslider on left panel x-axis; right panel follows via matches="x"
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.06)),
    )

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / f"{patient_id}_signals_post.html"
        fig.write_html(str(html_path))
        logger.info("Saved post-birth plot → %s", html_path)

    if show:
        fig.show()

    return fig


def plot_duration_distribution(
    per_patient_df: pd.DataFrame,
    output_dir: Path | None = None,
    show: bool = False,
) -> go.Figure:
    """Plot overlaid histograms of pre- and post-truncation signal durations.

    Args:
        per_patient_df: Per-patient stats DataFrame from build_summary().
        output_dir: Directory to write an interactive HTML file.  Skipped if None.
        show: Call fig.show() after building.

    Returns:
        Plotly Figure.
    """
    pre  = per_patient_df["duration_post_trim_min"].dropna()
    post = per_patient_df["duration_post_truncation_min"].dropna()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Post-trim duration (min)", "Post-truncation duration (min)"],
        horizontal_spacing=0.1,
    )

    for col_idx, (data, color, label) in enumerate([
        (pre,  _C["hr1_raw"],       "post-trim"),
        (post, _C["hr1_processed"], "post-truncation"),
    ], start=1):
        fig.add_trace(go.Histogram(
            x=data, name=label,
            marker_color=color, opacity=0.8,
            nbinsx=30, showlegend=False,
        ), row=1, col=col_idx)

        if not data.empty:
            med = float(data.median())
            fig.add_vline(
                x=med, line_dash="dash", line_color="black", line_width=1.5,
                annotation_text=f"median {med:.1f}",
                annotation_position="top right",
                annotation_font_size=10,
                row=1, col=col_idx,
            )

    fig.update_layout(
        title=dict(text="Signal Duration Distribution", font=dict(size=13), x=0.5, xanchor="center"),
        height=380,
        template="plotly_white",
    )
    fig.update_xaxes(title_text="minutes")
    fig.update_yaxes(title_text="count")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        html_path = output_dir / "duration_distribution.html"
        fig.write_html(str(html_path))
        logger.info("Saved duration plot → %s", html_path)

    if show:
        fig.show()

    return fig
