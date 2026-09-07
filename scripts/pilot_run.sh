#!/bin/bash
# Manual/laptop Kalshi-pilot run, driven by launchd (com.polymarketweather.kalshipilot) at 11:00
# local (≈ 15:00 UTC in summer — the capture-geometry hour the sigma tables are fitted at, and
# when tomorrow's lead-1 markets are open). DRY RUN by default.
#
# The daily-capture GitHub Action is the CANONICAL driver in both modes since 2026-09-07 (dry by
# default; live when the PILOT_LIVE repository variable is 1 — see the workflow's header). This
# script exists for a credentialed rehearsal or a one-off manual run: it never pushes, so rows it
# appends stay on this machine, and a live run here on a morning the Action also ran would place
# the same orders twice unless Kalshi's own resting-order/position dedupe catches them. Don't run
# both drivers live.
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
