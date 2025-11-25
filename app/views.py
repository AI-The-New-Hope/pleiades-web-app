"""Panel view constructors and layout helpers for the Gaia explorer app."""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict
from io import BytesIO
from typing import Optional

import pandas as pd
import panel as pn

from app.gaia import (
    GaiaQueryParams,
    clean_data,
    clear_query_cache,
    query_gaia,
)
from app.plots import (
    COLOR_FIELDS,
    make_3d_scatter,
    make_cmd_plot,
    make_parallax_histogram,
    make_placeholder_figure,
    make_proper_motion_plot,
    make_sky_scatter,
)


class GaiaQueryView:
    """Composite Panel view containing query controls and result panes."""

    def __init__(self) -> None:
        self.ra_input = pn.widgets.FloatInput(
            name="Right Ascension (deg)", value=56.75, step=0.1, start=-360.0, end=360.0
        )
        self.dec_input = pn.widgets.FloatInput(
            name="Declination (deg)", value=24.12, step=0.1, start=-90.0, end=90.0
        )
        self.radius_input = pn.widgets.FloatInput(
            name="Cone radius (deg)", value=2.5, step=0.1, start=0.1, end=10.0
        )
        self.parallax_slider = pn.widgets.RangeSlider(
            name="Parallax window (mas)", start=0.0, end=20.0, value=(4.0, 10.0), step=0.1
        )
        self.run_button = pn.widgets.Button(name="Run Gaia Query", button_type="primary")
        self.clear_cache_button = pn.widgets.Button(
            name="Clear Cached Results", button_type="warning"
        )
        self.color_select = pn.widgets.Select(
            name="Color by",
            options=list(COLOR_FIELDS.keys()),
            value="G magnitude",
        )
        self.color_select.param.watch(self._on_color_change, "value")
        self.download_button = pn.widgets.FileDownload(
            label="Download CSV",
            callback=self._download_csv,
            button_type="success",
            filename="gaia_query.csv",
            disabled=True,
        )
        self.run_button.on_click(self._on_run_query)
        self.clear_cache_button.on_click(self._on_clear_cache)

        self.spinner = pn.indicators.LoadingSpinner(value=False, width=40, height=40)
        self.status_text = pn.pane.Markdown("Ready to query Gaia.")
        self.alert = pn.pane.Alert(
            "",
            alert_type="success",
            visible=False,
            sizing_mode="stretch_width",
        )
        self.metrics = pn.widgets.DataFrame(
            pd.DataFrame(
                [
                    {
                        "Metric": "Query executed",
                        "Value": "—",
                    },
                    {
                        "Metric": "Rows (cleaned)",
                        "Value": "—",
                    },
                    {
                        "Metric": "Last parameters",
                        "Value": "—",
                    },
                ]
            ),
            disabled=True,
            width=400,
            height=150,
        )
        self.preview = pn.widgets.Tabulator(
            pd.DataFrame(),
            height=380,
            disabled=True,
            pagination="remote",
            page_size=20,
        )
        self.preview.visible = False

        placeholder = make_placeholder_figure("Run a Gaia query to populate plots.")
        self.sky_plot = pn.pane.Plotly(placeholder, height=420)
        self.scatter3d_plot = pn.pane.Plotly(placeholder, height=460)
        self.cmd_plot = pn.pane.Plotly(placeholder, height=420)
        self.proper_motion_plot = pn.pane.Plotly(placeholder, height=420)
        self.parallax_plot = pn.pane.Plotly(placeholder, height=400)
        self.tabs = pn.Tabs(
            ("Sky (2D)", self.sky_plot),
            ("Sky (3D)", self.scatter3d_plot),
            ("Color-Magnitude", self.cmd_plot),
            ("Proper Motion", self.proper_motion_plot),
            ("Parallax Histogram", self.parallax_plot),
        )

        self._last_clean_df: Optional[pd.DataFrame] = None
        self._last_params: Optional[GaiaQueryParams] = None
        self._last_run_time: Optional[dt.datetime] = None

    # ------------------------------------------------------------------
    # Layout helpers
    @property
    def sidebar(self) -> pn.Column:
        instructions = pn.pane.Markdown(
            """**Gaia DR3 Query Controls**  \\
            Adjust the cone-search parameters, then press **Run Gaia Query**.
            """
        )
        return pn.Column(
            instructions,
            self.ra_input,
            self.dec_input,
            self.radius_input,
            self.parallax_slider,
            pn.Row(self.run_button, self.spinner),
            self.clear_cache_button,
            sizing_mode="stretch_width",
        )

    @property
    def main(self) -> pn.Column:
        intro = pn.pane.Markdown(
            """### Query Status
            Use the controls on the left to download Gaia sources around the Pleiades
            (or any custom coordinates). Cleaned results will appear below for use in
            upcoming visualization steps.
            """,
            sizing_mode="stretch_width",
        )
        vis_controls = pn.Row(
            pn.pane.Markdown("**Visualization Controls**"),
            self.color_select,
            pn.Spacer(width=20),
            self.download_button,
            sizing_mode="stretch_width",
        )
        return pn.Column(
            intro,
            self.alert,
            self.status_text,
            pn.layout.HSpacer(height=10),
            pn.pane.Markdown("#### Query Metrics"),
            self.metrics,
            vis_controls,
            pn.pane.Markdown("#### Visualizations"),
            self.tabs,
            pn.pane.Markdown("#### Data Preview"),
            self.preview,
            sizing_mode="stretch_both",
        )

    # ------------------------------------------------------------------
    # Event handlers
    def _on_run_query(self, event) -> None:  # pragma: no cover - Panel runtime
        self.spinner.value = True
        self.alert.visible = False
        self.status_text.object = "Running Gaia query…"
        try:
            params = self._current_params()
        except ValueError as exc:  # validation failure from dataclass
            self.spinner.value = False
            self._show_alert(f"Parameter error: {exc}", "danger")
            return

        try:
            raw_df = query_gaia(params, use_cache=True)
            cleaned_df = clean_data(raw_df)
        except Exception as exc:  # network or TAP failure
            self.spinner.value = False
            self._show_alert(f"Gaia query failed: {exc}", "danger")
            return

        self._last_clean_df = cleaned_df
        self._last_params = params
        self._last_run_time = dt.datetime.utcnow()
        self.spinner.value = False

        self._update_metrics()
        self._update_preview()
        self._update_visualizations()
        self._show_alert(
            f"Retrieved {len(raw_df)} rows (cleaned: {len(cleaned_df)}).",
            "success",
        )
        self.status_text.object = "Query complete. Results cached for later plots."

    def _on_clear_cache(self, event) -> None:  # pragma: no cover - Panel runtime
        clear_query_cache()
        self._show_alert("Cleared cached Gaia results.", "warning")

    # ------------------------------------------------------------------
    def _current_params(self) -> GaiaQueryParams:
        start, end = self.parallax_slider.value
        return GaiaQueryParams(
            ra_deg=self.ra_input.value,
            dec_deg=self.dec_input.value,
            radius_deg=self.radius_input.value,
            parallax_min_mas=start,
            parallax_max_mas=end,
        )

    def _update_metrics(self) -> None:
        if not self._last_run_time or not self._last_params:
            return
        metrics_df = pd.DataFrame(
            [
                {
                    "Metric": "Query executed",
                    "Value": self._last_run_time.strftime("%Y-%m-%d %H:%M:%S UTC"),
                },
                {
                    "Metric": "Rows (cleaned)",
                    "Value": len(self._last_clean_df) if self._last_clean_df is not None else "0",
                },
                {
                    "Metric": "Last parameters",
                    "Value": asdict(self._last_params),
                },
            ]
        )
        self.metrics.value = metrics_df

    def _update_preview(self) -> None:
        if self._last_clean_df is None or self._last_clean_df.empty:
            self.preview.visible = False
            self.preview.value = pd.DataFrame()
            return
        head = self._last_clean_df.head(200)
        self.preview.value = head
        self.preview.visible = True

    def _update_visualizations(self) -> None:
        if self._last_clean_df is None or self._last_clean_df.empty:
            for pane in (
                self.sky_plot,
                self.scatter3d_plot,
                self.cmd_plot,
                self.proper_motion_plot,
                self.parallax_plot,
            ):
                pane.object = make_placeholder_figure(
                    "Run a Gaia query to populate plots."
                )
            self.download_button.disabled = True
            return

        color_field = COLOR_FIELDS[self.color_select.value]
        df = self._last_clean_df
        self.sky_plot.object = make_sky_scatter(df, color_field=color_field)
        self.scatter3d_plot.object = make_3d_scatter(df, color_field=color_field)
        self.cmd_plot.object = make_cmd_plot(df)
        self.proper_motion_plot.object = make_proper_motion_plot(
            df, color_field=color_field
        )
        self.parallax_plot.object = make_parallax_histogram(df)
        self.download_button.disabled = False

    def _download_csv(self) -> BytesIO:
        if self._last_clean_df is None or self._last_clean_df.empty:
            raise ValueError("No data available to download.")
        csv_bytes = self._last_clean_df.to_csv(index=False).encode("utf-8")
        return BytesIO(csv_bytes)

    def _on_color_change(self, event) -> None:  # pragma: no cover - Panel runtime
        if self._last_clean_df is None or self._last_clean_df.empty:
            return
        self._update_visualizations()

    def _show_alert(self, message: str, level: str) -> None:
        self.alert.object = message
        self.alert.alert_type = level
        self.alert.visible = True


__all__ = ["GaiaQueryView"]
