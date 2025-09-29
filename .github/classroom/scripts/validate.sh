#!/usr/bin/env bash
set -euo pipefail
test -f scripts/pleiades.py
test -f results/pleiades_scatter.png
test -f results/pleiades_cmd.png
test -f results/pleiades_histogram.png
git tag -l "v1.0" | grep -q "v1.0"
grep -Eiq "DBSCAN|HDBSCAN" scripts/pleiades.py
grep -Eiq "pages" README.md
echo "ALL TESTS PASSED"
