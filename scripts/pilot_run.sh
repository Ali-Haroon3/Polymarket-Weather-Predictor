#!/bin/bash
# Daily Kalshi-pilot run, driven by launchd (com.polymarketweather.kalshipilot) at 11:00 local
# (≈ 15:00 UTC in summer — the capture-geometry hour the sigma tables are fitted at, and when
# tomorrow's lead-1 markets are open). DRY RUN by default: add --live to the pilot line below only
# after the dry-run week checks out and the account is funded.
#
# Credentials come from the repo-root .env (gitignored, machine-local). This script never needs
# secrets of its own.
set -uo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 1

echo "=== $(date -u) ==="
# Fresh captures.jsonl (λ input) and code. ff-only: a dirty/diverged tree skips the update and the
# pilot still runs on what's here — never blocks on a merge prompt inside launchd.
git pull --ff-only origin main || echo "warning: git pull failed; running with local tree"
cargo build --release --bin kalshi_pilot || exit 1

./target/release/kalshi_pilot
