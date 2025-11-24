"""Panel entry point for the Gaia explorer app."""

from __future__ import annotations

import panel as pn

pn.extension("plotly")


def create_app() -> pn.template.Template:
    template = pn.template.FastListTemplate(
        title="Gaia Explorer",
        main=[pn.pane.Markdown("Phase 1 setup placeholder - content coming soon")],
    )
    return template


def main() -> None:
    create_app().servable()


if __name__.startswith("bokeh" ):
    # When served via `panel serve app/app.py`
    main()
