#!/usr/bin/env python3
"""Go-live gate for the favorite strategy: go_live_gate.py's unchanged rules plus one structural entry.

go_live_gate.py is hash-pinned by the scale-only challenger registration, so its STRUCTURAL table
cannot gain a "favorite" row before that trial ends. This wrapper imports it unmodified and adds
the row in memory. Every threshold, the fee model, the live-fill requirement and the --enforce
exit codes are the frozen file's own.

python3 scripts/favorite_gate.py --strategy favorite            # readout
python3 scripts/favorite_gate.py --strategy favorite --enforce  # exit 1 unless GO
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import go_live_gate as gate  # noqa: E402

gate.STRUCTURAL["favorite"] = (
    "favorite: lead-1 YES at ask 0.50-0.99, no fitted parameters; strategy-scoped drawdown and "
    "weekly-loss breakers in place since 2026-10-20"
)

if __name__ == "__main__":
    sys.exit(gate.main())
