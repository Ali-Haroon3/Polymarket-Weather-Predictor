#!/bin/bash
# Daily forward-capture + dashboard refresh, driven by launchd (com.polymarketweather.dailycapture).
# Snapshots active weather markets, finalizes outcomes as they resolve, and regenerates dashboard.html.
# No `set -e`: capture_prices may exit non-zero on a transient fetch failure, and we still want the
# dashboard to re-render from the last-good captures.jsonl.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1

echo "=== $(date) ==="
./target/release/capture_prices
./target/release/weather_dashboard --output dashboard.html
