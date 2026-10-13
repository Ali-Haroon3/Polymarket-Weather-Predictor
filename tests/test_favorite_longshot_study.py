"""The frozen favorite rule selects only 0.50 <= ask < 0.99, inside its capture window, after fees."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import favorite_longshot_study as study


def row(ask, outcome, cap="2026-10-13", ticker="T"):
    return dict(source="kalshi", market_id=f"{ticker}-{ask}-{cap}", city="NYC", captured_at=cap,
                target_date="2026-10-14", best_bid=None, best_ask=ask, outcome=outcome)


class FavoriteRule(unittest.TestCase):
    def test_selection_window_and_settlement(self):
        rows = [row(0.49, 1), row(0.50, 1), row(0.62, 0), row(0.99, 1),
                row(0.55, None), row(0.70, 1, cap="2026-10-12")]
        trades, unresolved = study.favorite_trades(rows, study.FORWARD_START, study.FORWARD_END)
        self.assertEqual([t["price"] for t in trades], [0.50, 0.62])
        self.assertEqual(unresolved, 1)
        win, loss = trades
        # 29 contracts at 50c = $14.50 + ceil(0.07*29*0.25) = $0.51 fee: $15.01 > $15, so 28 contracts.
        self.assertEqual(win["contracts"], 28)
        self.assertAlmostEqual(win["net"], 28 - win["cost"])
        self.assertAlmostEqual(loss["net"], -loss["cost"])
        self.assertLessEqual(max(t["cost"] for t in trades), 15.0)


if __name__ == "__main__":
    unittest.main()
