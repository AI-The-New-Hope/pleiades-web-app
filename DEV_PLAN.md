# Gaia Web App Development Plan

## Overview
Build a local-first interactive Panel application that queries Gaia DR3 via `astroquery`, visualizes the data with Plotly (including 3D scenes), and later expands with clustering tools plus optional FastAPI mounting for API integrations and remote hosting.

## Goals & Non-Goals
- **Goals**
  - Support user-driven Gaia cone-search queries with adjustable RA/Dec, radius, and parallax filters.
  - Render visually appealing 2D/3D Plotly views with hover metadata and color mapping toggles.
  - Cache and reuse Gaia results to stay within archive rate limits.
  - Stage future enhancements: DBSCAN clustering controls and FastAPI embedding.
- **Non-Goals (initially)**
  - Cloud deployment or authentication.
  - Supporting arbitrary Gaia tables beyond `gaiadr3.gaia_source`.

## Assumptions & Constraints
- App runs locally (Panel server) but should avoid hard-coded ports to ease later FastAPI mounting.
- Python >=3.10 with `astroquery`, `panel`, `plotly`, `pandas`, `numpy`, `scikit-learn` already available or installable.
- Gaia query latency (10–60 s) requires responsive UI cues (loading spinners, disabled buttons).
- Results held in-memory; dataset size limited by cone search filters.

## Phase 1 — Query & Visualization (current focus)
1. **Environment Setup**
   - Create `app/` package with `__init__.py`, `gaia.py`, `views.py`, `app.py` (Panel entry).
   - Add `requirements.txt` or `pyproject` extras for Panel/Plotly.
2. **Data Layer**
   - Move `query_gaia`, `clean_data`, and helpers from `scripts/pleiades.py` into `app/gaia.py` (keeping CLI script functional by importing shared code).
   - Implement `GaiaQueryParams` dataclass (RA, Dec, radius, parallax range, optional magnitude filters) plus validation (degrees, mas ranges).
   - Add caching decorator (e.g., `functools.lru_cache` keyed by frozen params) and manual cache invalidation button to avoid stale data.
3. **Panel Widgets & State**
   - Build sidebar controls: numeric inputs for RA/Dec/radius, sliders for parallax min/max, optional magnitude filter, “Run Query” button, and status panel.
   - Tie widgets to param/state objects (either `param.Parameterized` class or `pn.state.cache + pn.bind`).
   - Provide basic telemetry (last query time, row count, warnings if rows > threshold).
4. **Visualization Layer**
   - Create `make_sky_scatter(df)` using `px.scatter` (RA vs Dec) with magnitude color map.
   - Create `make_3d_spatial(df)` using `px.scatter_3d` (RA, Dec, parallax or Cartesian transform) with color selector dropdown (magnitude, parallax, bp_rp).
   - Create additional tabs: CMD plot (`bp_rp` vs `phot_g_mean_mag`), proper motion chart (`pmra` vs `pmdec`), histogram of parallax.
   - Add data table preview (Panel DataFrame pane) with download button (CSV) per current filter state.
5. **User Feedback & UX**
   - Wrap long operations with `pn.indicators.LoadingSpinner` or `pn.state.busy_indicators`.
   - Surface query errors (network, invalid ADQL) via `pn.alert` components.
   - Provide instructions on first tab describing input ranges and usage tips.
6. **Testing & Validation**
   - Add unit tests for `GaiaQueryParams` validation and caching key logic (pytest).
  - Manual verification: run Panel server (`panel serve app/main.py --autoreload`), execute multiple parameter combinations, ensure plots update.

## Phase 2 — Clustering & Advanced Analysis
1. **DBSCAN Integration**
   - Port `run_dbscan_clustering` into shared module; ensure configurable hyperparameters (eps, min_samples).
   - Expose clustering toggle + hyperparameter controls; when enabled, recompute clusters on filtered data.
2. **UI Enhancements**
   - Add cluster-aware legends, color maps, and summary stats (counts per cluster, % noise).
   - Provide selection tools (e.g., `pn.widgets.MultiSelect`) to highlight specific clusters across plots.
3. **Performance Considerations**
   - Split data pipeline so clustering runs on a background thread or `pn.state.execute` to keep UI responsive.
   - Implement progress indicators when clustering large samples (>20k rows).
4. **Data Export**
   - Allow downloading clustered subsets as CSV/Parquet, embedding query parameters + clustering settings in metadata.

## Phase 3 — FastAPI Mounting & API Surface
1. **Service Architecture**
   - Create `main.py` with FastAPI app; mount Panel at `/app` using `FastAPI.mount` + `panel.template.Template`.
   - Provide REST endpoints:
     - `GET /healthz` (simple status)
     - `POST /gaia/query` accepting JSON params, returning JSON payload or signed URL to cached CSV.
2. **Auth & Security (placeholder)**
   - Design hooks for API tokens or OAuth if remote deployment becomes necessary.
3. **Deployment Scripts**
   - Add `uvicorn main:app --reload` dev command, `Dockerfile` for future containerization, and `panel serve` instructions for standalone mode.

## Phase 4 — Polishing, Docs, and Future Ideas
- Document usage in `README` (screenshots, how to run Panel-only vs FastAPI-mounted versions).
- Add scripted regression tests: mock Gaia responses (via `astroquery.gaia.Gaia.launch_job`) to avoid network reliance in CI.
- Explore advanced visualization libraries (Deck.gl via `pydeck-pane`, PyVista) if Plotly 3D becomes limiting.
- Consider integrating NASA/IPAC Firefly or ESA’s Gaia Archive API for cross-links.

## Deliverables & Milestones
- **M1 (Phase 1 complete):** Fully functional local Panel app with adjustable query form and core Plotly visualizations, plus caching + CSV export.
- **M2 (Phase 2 complete):** Cluster controls live, cluster-aware visualizations, advanced exports.
- **M3 (Phase 3 complete):** FastAPI service hosting both REST endpoints and embedded Panel UI, ready for optional cloud deployment.

## Open Questions for Later
1. Should we transform coordinates to Cartesian XYZ for 3D plots (requires astropy units)?
2. Do we need persistence of past queries (SQLite, parquet cache) or is in-memory enough?
3. What authentication model (if any) is required once FastAPI endpoints exist?
