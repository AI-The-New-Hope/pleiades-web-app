#!/usr/bin/env python3
"""Query Gaia DR3 around the Pleiades, cluster likely members, and create plots."""

from __future__ import annotations

import logging
from itertools import cycle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
from astroquery.gaia import Gaia
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Coordinates of the Pleiades cluster (J2000)
PLEIADES_RA_DEG = 56.75
PLEIADES_DEC_DEG = 24.12
# Cone-search radius (degrees)
SEARCH_RADIUS_DEG = 2.5
# Parallax window (milliarcseconds) to limit the download volume
PARALLAX_MIN_MAS = 4.0
PARALLAX_MAX_MAS = 10.0

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

# Output locations
RESULTS_DIR = Path("results")
SCATTER_PLOT_PATH = RESULTS_DIR / "pleiades_scatter.png"
CMD_PLOT_PATH = RESULTS_DIR / "pleiades_cmd.png"
PARALLAX_HIST_PATH = RESULTS_DIR / "pleiades_histogram.png"
CLUSTER_SKY_PATH = RESULTS_DIR / "pleiades_cluster_sky.png"
PROPER_MOTION_PLOT_PATH = RESULTS_DIR / "pleiades_proper_motion.png"

# DBSCAN parameters (tunings can be adjusted later if needed)
DBSCAN_EPS = 0.12
DBSCAN_MIN_SAMPLES = 20

# Colour palette for cluster visualisations
CLUSTER_COLOR_SEQUENCE = (
    px.colors.qualitative.Safe
    + px.colors.qualitative.Bold
    + px.colors.qualitative.Dark24
)

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
        AND parallax BETWEEN {PARALLAX_MIN_MAS} AND {PARALLAX_MAX_MAS}
    """.strip()


def query_gaia() -> pd.DataFrame:
    """Run the Gaia query and return the results as a DataFrame."""
    logging.info(
        "Querying Gaia DR3 around RA=%s°, Dec=%s° with radius %.1f° and parallax in [%.1f, %.1f] mas",
        PLEIADES_RA_DEG,
        PLEIADES_DEC_DEG,
        SEARCH_RADIUS_DEG,
        PARALLAX_MIN_MAS,
        PARALLAX_MAX_MAS,
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


def standardize_features(df: pd.DataFrame, columns: Iterable[str]) -> tuple[pd.Index, np.ndarray]:
    """Standardize selected columns and return mask plus scaled feature matrix."""
    mask = df[list(columns)].notnull().all(axis=1)
    if not mask.any():
        return df.index[mask], np.empty((0, len(columns)))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.loc[mask, columns])
    return df.index[mask], scaled


def run_dbscan_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """Cluster sources using DBSCAN on kinematic features."""
    kinematics_cols = ["pmra", "pmdec", "parallax"]
    valid_index, features = standardize_features(df, kinematics_cols)
    df = df.copy()
    df["cluster_label"] = -1
    df["is_cluster_member"] = False
    if features.size == 0:
        logging.warning("No rows contained all kinematic features; skipping clustering.")
        return df

    logging.info(
        "Running DBSCAN with eps=%s, min_samples=%s on %d stars",
        DBSCAN_EPS,
        DBSCAN_MIN_SAMPLES,
        features.shape[0],
    )
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES)
    labels = clustering.fit_predict(features)

    df.loc[valid_index, "cluster_label"] = labels
    df.loc[valid_index, "is_cluster_member"] = labels >= 0

    unique_labels, counts = np.unique(labels, return_counts=True)
    cluster_counts = {
        int(label): int(count)
        for label, count in zip(unique_labels, counts, strict=True)
    }
    member_total = sum(count for label, count in cluster_counts.items() if label != -1)
    noise_total = cluster_counts.get(-1, 0)
    n_clusters = len([label for label in unique_labels if label != -1])
    logging.info(
        "DBSCAN identified %d cluster(s) with %d candidate members; %d stars tagged as noise",
        n_clusters,
        member_total,
        noise_total,
    )
    return df


def cluster_category_series(df: pd.DataFrame) -> pd.Series:
    """Return a categorical series labelling clusters vs noise for plotting."""
    labels = df.get("cluster_label")
    if labels is None:
        return pd.Series(["Noise"] * len(df), index=df.index)
    labels = labels.fillna(-1).astype(int)
    categories = labels.astype(str)
    categories = categories.where(labels >= 0, other="Noise")
    return categories


def build_cluster_color_map(categories: Iterable[str]) -> dict[str, str]:
    """Create a discrete colour map for cluster categories."""
    color_map = {"Noise": "#7f7f7f"}
    unique_clusters = sorted({int(label) for label in categories if label != "Noise"})
    palette_cycle = cycle(CLUSTER_COLOR_SEQUENCE)
    for cluster_id in unique_clusters:
        color_map[str(cluster_id)] = next(palette_cycle)
    return color_map


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
            "cluster_label": True if "cluster_label" in df.columns else False,
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


def create_cluster_sky_plot(df: pd.DataFrame) -> None:
    """Plot the sky distribution coloured by DBSCAN cluster label."""
    sky_df = df.copy()
    sky_df["cluster_category"] = cluster_category_series(sky_df)
    color_map = build_cluster_color_map(sky_df["cluster_category"]) if not sky_df.empty else {}
    logging.info("Rendering cluster-labelled sky plot to %s", CLUSTER_SKY_PATH)
    fig = px.scatter(
        sky_df,
        x="ra",
        y="dec",
        color="cluster_category",
        color_discrete_map=color_map,
        hover_data={
            "source_id": True,
            "cluster_label": True,
            "is_cluster_member": True,
            "parallax": ":.3f",
            "pmra": ":.3f",
            "pmdec": ":.3f",
        },
        labels={
            "ra": "Right Ascension (deg)",
            "dec": "Declination (deg)",
            "cluster_category": "Cluster label",
        },
        title="DBSCAN Clusters on the Sky",
    )
    fig.update_traces(marker=dict(size=5, opacity=0.7))
    fig.write_image(str(CLUSTER_SKY_PATH), width=1000, height=800, scale=2)
    logging.info("Saved cluster sky plot with %d points", len(sky_df))


def create_proper_motion_plot(df: pd.DataFrame) -> None:
    """Plot proper motions coloured by DBSCAN cluster label."""
    motion_df = df.dropna(subset=["pmra", "pmdec"]).copy()
    if motion_df.empty:
        logging.warning("No proper motion data available for cluster plot.")
        return
    motion_df["cluster_category"] = cluster_category_series(motion_df)
    color_map = build_cluster_color_map(motion_df["cluster_category"])
    logging.info("Rendering proper motion plot to %s", PROPER_MOTION_PLOT_PATH)
    fig = px.scatter(
        motion_df,
        x="pmra",
        y="pmdec",
        color="cluster_category",
        color_discrete_map=color_map,
        hover_data={
            "source_id": True,
            "cluster_label": True,
            "is_cluster_member": True,
            "parallax": ":.3f",
            "phot_g_mean_mag": ":.2f",
            "bp_rp": ":.2f",
        },
        labels={
            "pmra": "Proper motion in RA (mas/yr)",
            "pmdec": "Proper motion in Dec (mas/yr)",
            "cluster_category": "Cluster label",
        },
        title="DBSCAN Clusters in Proper Motion Space",
    )
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.write_image(str(PROPER_MOTION_PLOT_PATH), width=900, height=900, scale=2)
    logging.info("Saved proper motion plot with %d points", len(motion_df))


def create_cmd_plot(df: pd.DataFrame) -> None:
    """Generate a color-magnitude diagram highlighting cluster members."""
    cmd_df = df.dropna(subset=["bp_rp", "phot_g_mean_mag"]).copy()
    if cmd_df.empty:
        logging.warning("No data available for CMD plot.")
        return
    logging.info("Rendering CMD plot to %s", CMD_PLOT_PATH)
    fig = px.scatter(
        cmd_df,
        x="bp_rp",
        y="phot_g_mean_mag",
        color="is_cluster_member",
        color_discrete_map={True: "#d62728", False: "#7f7f7f"},
        hover_data={
            "source_id": True,
            "parallax": ":.3f",
            "pmra": ":.3f",
            "pmdec": ":.3f",
            "phot_g_mean_mag": ":.2f",
            "bp_rp": ":.2f",
            "cluster_label": True,
        },
        labels={
            "bp_rp": "BP - RP (mag)",
            "phot_g_mean_mag": "G Magnitude",
            "is_cluster_member": "Cluster member",
        },
        title="Pleiades Color-Magnitude Diagram",
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_traces(marker=dict(size=6, opacity=0.8))
    fig.write_image(str(CMD_PLOT_PATH), width=1000, height=800, scale=2)
    logging.info("Saved CMD plot with %d points", len(cmd_df))


def create_parallax_histogram(df: pd.DataFrame) -> None:
    """Plot a parallax distribution comparing members and field stars."""
    hist_df = df.dropna(subset=["parallax"]).copy()
    if hist_df.empty:
        logging.warning("No parallax data available for histogram.")
        return
    logging.info("Rendering parallax histogram to %s", PARALLAX_HIST_PATH)
    fig = px.histogram(
        hist_df,
        x="parallax",
        color="is_cluster_member",
        nbins=60,
        barmode="overlay",
        opacity=0.75,
        color_discrete_map={True: "#d62728", False: "#7f7f7f"},
        labels={
            "parallax": "Parallax (mas)",
            "is_cluster_member": "Cluster member",
        },
        title="Parallax Distribution Near the Pleiades",
    )
    fig.update_layout(legend=dict(title=""))
    fig.write_image(str(PARALLAX_HIST_PATH), width=1000, height=800, scale=2)
    logging.info("Saved parallax histogram with %d total stars", len(hist_df))


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = query_gaia()
    cleaned = clean_data(df)
    if cleaned.empty:
        logging.warning("No data remaining after cleaning; skipping plots.")
        return
    clustered = run_dbscan_clustering(cleaned)
    create_scatter_plot(clustered)
    create_cluster_sky_plot(clustered)
    create_proper_motion_plot(clustered)
    create_cmd_plot(clustered)
    create_parallax_histogram(clustered)


if __name__ == "__main__":
    main()
