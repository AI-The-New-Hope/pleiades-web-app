#!/usr/bin/env python3
"""Query Gaia DR3 around the Pleiades and create an RA-Dec scatter plot."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
from astroquery.gaia import Gaia

# Coordinates of the Pleiades cluster (J2000)
PLEIADES_RA_DEG = 56.75
PLEIADES_DEC_DEG = 24.12
# Cone-search radius (degrees)
SEARCH_RADIUS_DEG = 2.5

# Columns we need from Gaia
GAIA_COLUMNS = [
    "source_id",
    "ra",
    "dec",
    "parallax",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "bp_rp",
]

RESULTS_DIR = Path("results")
SCATTER_PLOT_PATH = RESULTS_DIR / "pleiades_scatter.png"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def build_gaia_query() -> str:
    """Compose an ADQL cone-search query around the Pleiades."""
    columns = ", ".join(GAIA_COLUMNS)
    return f"""
        SELECT {columns}
        FROM gaiadr3.gaia_source
        WHERE 1 = CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {PLEIADES_RA_DEG}, {PLEIADES_DEC_DEG}, {SEARCH_RADIUS_DEG})
        )
    """.strip()


def query_gaia() -> pd.DataFrame:
    """Run the Gaia query and return the results as a DataFrame."""
    logging.info(
        "Querying Gaia DR3 around RA=%s°, Dec=%s° with radius %.1f°",
        PLEIADES_RA_DEG,
        PLEIADES_DEC_DEG,
        SEARCH_RADIUS_DEG,
    )
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    query = build_gaia_query()
    job = Gaia.launch_job_async(query, dump_to_file=False)
    results_table = job.get_results()
    df = results_table.to_pandas()
    logging.info("Retrieved %d rows from Gaia", len(df))
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing key columns and report basic stats."""
    required_columns = [c for c in GAIA_COLUMNS if c in df.columns]
    cleaned = df.dropna(subset=required_columns)
    dropped = len(df) - len(cleaned)
    if dropped:
        logging.info("Dropped %d rows with missing values", dropped)
    logging.info(
        "Parallax median: %.3f mas; Proper motion (pmra, pmdec) medians: %.3f, %.3f mas/yr",
        cleaned["parallax"].median(),
        cleaned["pmra"].median(),
        cleaned["pmdec"].median(),
    )
    return cleaned


def create_scatter_plot(df: pd.DataFrame) -> None:
    """Create and save an RA-Dec scatter plot colored by G magnitude."""
    logging.info("Rendering RA-Dec scatter plot to %s", SCATTER_PLOT_PATH)
    fig = px.scatter(
        df,
        x="ra",
        y="dec",
        color="phot_g_mean_mag",
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
            "phot_g_mean_mag": "G Magnitude",
        },
        title="Gaia DR3 Sources Near the Pleiades",
    )
    fig.update_traces(marker=dict(size=5, opacity=0.6))
    fig.update_layout(coloraxis_colorbar_title="G mag")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(SCATTER_PLOT_PATH), width=1000, height=800, scale=2)
    logging.info("Saved scatter plot with %d points", len(df))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = query_gaia()
    cleaned = clean_data(df)
    if cleaned.empty:
        logging.warning("No data remaining after cleaning; skipping plot.")
        return
    create_scatter_plot(cleaned)


if __name__ == "__main__":
    main()
