#!/bin/bash
# Daily forward-capture + dashboard refresh, driven by cron or launchd.
# Snapshots active weather markets, finalizes outcomes as they resolve, and regenerates dashboard.html.
# No `set -e`: capture_prices may exit non-zero on a transient fetch failure, and we still want the
# dashboard to re-render from the last-good captures.jsonl.
set -uo pipefail

# Cron and launchd do not load the interactive shell's Cargo PATH.
export PATH="${HOME}/.cargo/bin:${PATH:-/usr/bin:/bin}"

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1

echo "=== $(date) ==="
if ! cargo build --release --bin capture_prices --bin weather_dashboard; then
    echo "error: build failed; refusing to run stale capture or dashboard binaries" >&2
    exit 1
fi
./target/release/capture_prices
./target/release/weather_dashboard --output dashboard.html
