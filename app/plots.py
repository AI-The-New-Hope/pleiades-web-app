"""Plotly figure factories for the Gaia explorer Panel app."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Public constants ---------------------------------------------------------

COLOR_FIELDS = {
    "G magnitude": "phot_g_mean_mag",
    "Parallax": "parallax",
    "BP-RP color": "bp_rp",
}


def _clean_subset(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    subset = df.dropna(subset=list(columns))
    return subset


def make_placeholder_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        showarrow=False,
        font=dict(size=16),
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=400)
    return fig


def make_sky_scatter(df: pd.DataFrame, *, color_field: str) -> go.Figure:
    required = {"ra", "dec", color_field}
    subset = _clean_subset(df, required)
    if subset.empty:
        return make_placeholder_figure("No data available for sky scatter plot.")
    fig = px.scatter(
        subset,
        x="ra",
        y="dec",
        color=color_field,
        color_continuous_scale="Viridis_r",
        hover_data={
            "source_id": True,
            "parallax": ":.3f",
            "pmra": ":.3f",
            "pmdec": ":.3f",
            "phot_g_mean_mag": ":.2f",
            "bp_rp": ":.2f",
        },
        labels={
            "ra": "Right Ascension (deg)",
            "dec": "Declination (deg)",
            color_field: color_field.replace("_", " ").title(),
        },
        title="Gaia DR3 Sources (RA vs Dec)",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.65))
    fig.update_layout(height=420, coloraxis_colorbar_title=color_field)
    return fig


def make_3d_scatter(df: pd.DataFrame, *, color_field: str) -> go.Figure:
    required = {"ra", "dec", "parallax", color_field}
    subset = _clean_subset(df, required)
    if subset.empty:
        return make_placeholder_figure("No data available for 3D scatter plot.")
    fig = px.scatter_3d(
        subset,
        x="ra",
        y="dec",
        z="parallax",
        color=color_field,
        color_continuous_scale="Viridis_r",
        hover_data={
            "source_id": True,
            "parallax": ":.3f",
            "pmra": ":.3f",
            "pmdec": ":.3f",
            "phot_g_mean_mag": ":.2f",
            "bp_rp": ":.2f",
        },
        labels={
            "ra": "RA (deg)",
            "dec": "Dec (deg)",
            "parallax": "Parallax (mas)",
            color_field: color_field.replace("_", " ").title(),
        },
        title="3D Scatter (RA, Dec, Parallax)",
    )
    fig.update_traces(marker=dict(size=3, opacity=0.8))
    fig.update_layout(height=460, coloraxis_colorbar_title=color_field)
    return fig


def make_cmd_plot(df: pd.DataFrame) -> go.Figure:
    required = {"bp_rp", "phot_g_mean_mag"}
    subset = _clean_subset(df, required)
    if subset.empty:
        return make_placeholder_figure("No data available for color-magnitude diagram.")
    fig = px.scatter(
        subset,
        x="bp_rp",
        y="phot_g_mean_mag",
        color="phot_g_mean_mag",
        color_continuous_scale="Plasma_r",
        hover_data={
            "source_id": True,
            "parallax": ":.3f",
            "pmra": ":.3f",
            "pmdec": ":.3f",
        },
        labels={
            "bp_rp": "BP - RP (mag)",
            "phot_g_mean_mag": "G Magnitude",
        },
        title="Color-Magnitude Diagram",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.update_layout(height=420)
    return fig


def make_proper_motion_plot(df: pd.DataFrame, *, color_field: str) -> go.Figure:
    required = {"pmra", "pmdec", color_field}
    subset = _clean_subset(df, required)
    if subset.empty:
        return make_placeholder_figure("No data available for proper motion plot.")
    fig = px.scatter(
        subset,
        x="pmra",
        y="pmdec",
        color=color_field,
        color_continuous_scale="Cividis",
        hover_data={
            "source_id": True,
            "parallax": ":.3f",
            "phot_g_mean_mag": ":.2f",
        },
        labels={
            "pmra": "pmRA (mas/yr)",
            "pmdec": "pmDec (mas/yr)",
            color_field: color_field.replace("_", " ").title(),
        },
        title="Proper Motion Space",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.75))
    fig.update_layout(height=420)
    return fig


def make_parallax_histogram(df: pd.DataFrame) -> go.Figure:
    required = {"parallax"}
    subset = _clean_subset(df, required)
    if subset.empty:
        return make_placeholder_figure("No data available for parallax histogram.")
    fig = px.histogram(
        subset,
        x="parallax",
        nbins=60,
        opacity=0.8,
        labels={"parallax": "Parallax (mas)"},
        title="Parallax Distribution",
        color_discrete_sequence=["#1f77b4"],
    )
    fig.update_layout(height=400)
    return fig


__all__ = [
    "COLOR_FIELDS",
    "make_placeholder_figure",
    "make_cmd_plot",
    "make_parallax_histogram",
    "make_proper_motion_plot",
    "make_sky_scatter",
    "make_3d_scatter",
]
