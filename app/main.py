"""Panel entry point for the Gaia explorer app."""

from __future__ import annotations

from pathlib import Path
import sys

import panel as pn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.views import GaiaQueryView

pn.extension("plotly", "tabulator", sizing_mode="stretch_width")


def create_app() -> pn.template.Template:
    query_view = GaiaQueryView()
    template = pn.template.FastListTemplate(
        title="Gaia Explorer",
        sidebar=[query_view.sidebar],
        main=[query_view.main],
    )
    return template


def main() -> None:
    create_app().servable()


if __name__.startswith("bokeh"):
    # When served via `panel serve app/main.py`
    main()
