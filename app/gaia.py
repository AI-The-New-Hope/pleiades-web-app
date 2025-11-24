"""Gaia querying, cleaning, and clustering utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from astroquery.gaia import Gaia
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Default cone-search parameters around the Pleiades cluster.
GAIA_COLUMNS: tuple[str, ...] = (
    "source_id",
    "ra",
    "dec",
    "parallax",
    "pmra",
    "pmdec",
    "phot_g_mean_mag",
    "bp_rp",
)

DBSCAN_EPS = 0.12
DBSCAN_MIN_SAMPLES = 20


@dataclass(frozen=True)
class GaiaQueryParams:
    """Validated parameter set for a Gaia cone search."""

    ra_deg: float = 56.75
    dec_deg: float = 24.12
    radius_deg: float = 2.5
    parallax_min_mas: float = 4.0
    parallax_max_mas: float = 10.0

    def __post_init__(self) -> None:
        if not (-360.0 <= self.ra_deg <= 360.0):  # wrap-around safety
            raise ValueError("ra_deg must be within [-360, 360]")
        if not (-90.0 <= self.dec_deg <= 90.0):
            raise ValueError("dec_deg must be within [-90, 90]")
        if not (0.0 < self.radius_deg <= 10.0):
            raise ValueError("radius_deg must be within (0, 10] degrees")
        if not (0.0 <= self.parallax_min_mas < self.parallax_max_mas):
            raise ValueError("parallax_min_mas must be < parallax_max_mas and >= 0")

    @property
    def identifier(self) -> tuple[float, float, float, float, float]:
        """Hash-friendly tuple representing this parameter set."""

        return (
            round(self.ra_deg, 8),
            round(self.dec_deg, 8),
            round(self.radius_deg, 8),
            round(self.parallax_min_mas, 8),
            round(self.parallax_max_mas, 8),
        )


DEFAULT_QUERY_PARAMS = GaiaQueryParams()


def build_gaia_query(
    params: GaiaQueryParams, columns: Sequence[str] | None = None
) -> str:
    """Compose an ADQL cone-search query for the supplied parameters."""

    cols = ", ".join(columns or GAIA_COLUMNS)
    return (
        "SELECT {columns} FROM gaiadr3.gaia_source WHERE 1 = CONTAINS("
        "POINT('ICRS', ra, dec),"
        "CIRCLE('ICRS', {ra}, {dec}, {radius})) "
        "AND parallax BETWEEN {pmin} AND {pmax}"
    ).format(
        columns=cols,
        ra=params.ra_deg,
        dec=params.dec_deg,
        radius=params.radius_deg,
        pmin=params.parallax_min_mas,
        pmax=params.parallax_max_mas,
    )


def _run_gaia_query(params: GaiaQueryParams, columns: tuple[str, ...]) -> pd.DataFrame:
    logger.info(
        "Querying Gaia DR3 around RA=%s°, Dec=%s° (radius %.2f°) with parallax in [%.2f, %.2f] mas",
        params.ra_deg,
        params.dec_deg,
        params.radius_deg,
        params.parallax_min_mas,
        params.parallax_max_mas,
    )
    Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
    query = build_gaia_query(params, columns)
    job = Gaia.launch_job_async(query, dump_to_file=False)
    results_table = job.get_results()
    df = results_table.to_pandas()
    logger.info("Retrieved %d rows from Gaia", len(df))
    return df


@lru_cache(maxsize=16)
def _cached_gaia_query(params: GaiaQueryParams, columns: tuple[str, ...]) -> pd.DataFrame:
    """Cached Gaia query to avoid repeated TAP calls for the same parameter set."""

    return _run_gaia_query(params, columns)


def clear_query_cache() -> None:
    """Reset the Gaia query cache (useful for long-running Panel sessions)."""

    _cached_gaia_query.cache_clear()


def query_gaia(
    params: GaiaQueryParams | None = None,
    *,
    columns: Sequence[str] | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Run the Gaia query and return results as a DataFrame."""

    resolved_params = params or DEFAULT_QUERY_PARAMS
    resolved_columns = tuple(columns or GAIA_COLUMNS)
    if use_cache:
        df = _cached_gaia_query(resolved_params, resolved_columns)
    else:
        df = _run_gaia_query(resolved_params, resolved_columns)
    # Return a copy so callers can mutate without affecting the cache.
    return df.copy(deep=True)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows missing required Gaia columns and log summary stats."""

    required_columns = [c for c in GAIA_COLUMNS if c in df.columns]
    cleaned = df.dropna(subset=required_columns)
    dropped = len(df) - len(cleaned)
    if dropped:
        logger.info("Dropped %d rows with missing values", dropped)
    if not cleaned.empty:
        logger.info(
            "Parallax median: %.3f mas; Proper motion medians (pmra, pmdec): %.3f, %.3f mas/yr",
            cleaned["parallax"].median(),
            cleaned["pmra"].median(),
            cleaned["pmdec"].median(),
        )
    return cleaned


def standardize_features(
    df: pd.DataFrame, columns: Iterable[str]
) -> tuple[pd.Index, np.ndarray]:
    """Standardize selected columns and return mask + scaled features."""

    mask = df[list(columns)].notnull().all(axis=1)
    if not mask.any():
        return df.index[mask], np.empty((0, len(columns)))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.loc[mask, columns])
    return df.index[mask], scaled


def run_dbscan_clustering(
    df: pd.DataFrame,
    *,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
) -> pd.DataFrame:
    """Cluster sources using DBSCAN on kinematic features."""

    kinematics_cols = ("pmra", "pmdec", "parallax")
    valid_index, features = standardize_features(df, kinematics_cols)
    df = df.copy()
    df["cluster_label"] = -1
    df["is_cluster_member"] = False
    if features.size == 0:
        logger.warning("No rows contained all kinematic features; skipping clustering.")
        return df

    logger.info(
        "Running DBSCAN with eps=%s, min_samples=%s on %d stars",
        eps,
        min_samples,
        features.shape[0],
    )
    clustering = DBSCAN(eps=eps, min_samples=min_samples)
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
    logger.info(
        "DBSCAN identified %d cluster(s) with %d candidate members; %d stars tagged as noise",
        n_clusters,
        member_total,
        noise_total,
    )
    return df


__all__ = [
    "DBSCAN_EPS",
    "DBSCAN_MIN_SAMPLES",
    "DEFAULT_QUERY_PARAMS",
    "GaiaQueryParams",
    "GAIA_COLUMNS",
    "build_gaia_query",
    "clean_data",
    "clear_query_cache",
    "query_gaia",
    "run_dbscan_clustering",
    "standardize_features",
]
