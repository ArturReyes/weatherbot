import unittest
from datetime import datetime, timezone

from trading_risk import RiskLimits, assess_trade_risk


NOW = datetime(2026, 7, 8, 12, tzinfo=timezone.utc).timestamp()
LIMITS = RiskLimits(
    max_total_exposure_pct=0.25,
    max_event_exposure_pct=0.10,
    max_daily_loss_pct=0.05,
    max_open_positions=5,
    max_signal_age_seconds=120,
)


def decide(state, *, size=10, city="nyc", date="2026-07-09", created=NOW):
    return assess_trade_risk(
        state,
        size_usdc=size,
        city_slug=city,
        date_str=date,
        signal_created_at=created,
        bankroll=100,
        limits=LIMITS,
        now_ts=NOW,
    )


class PortfolioRiskTests(unittest.TestCase):
    def test_rejects_projected_total_exposure_breach(self):
        state = {"positions": [{"status": "open", "amount": 20}]}
        self.assertEqual(decide(state).reason, "total_exposure_limit")

    def test_ambiguous_requested_amount_counts_as_exposure(self):
        state = {"positions": [{"status": "unknown", "requested_amount": 20, "amount": 0}]}
        self.assertEqual(decide(state).reason, "total_exposure_limit")

    def test_rejects_projected_same_event_exposure_breach(self):
        state = {"positions": [{
            "status": "open", "amount": 5, "city_slug": "nyc", "date": "2026-07-09"
        }]}
        self.assertEqual(decide(state, size=6).reason, "event_exposure_limit")

    def test_rejects_realized_daily_loss_breach(self):
        state = {"positions": [{
            "status": "closed", "pnl": -5.0, "exited_at": "2026-07-08T03:00:00+00:00"
        }]}
        self.assertEqual(decide(state, size=1).reason, "daily_loss_limit")

    def test_rejects_maximum_active_positions(self):
        state = {"positions": [{"status": "open", "amount": 1} for _ in range(5)]}
        self.assertEqual(decide(state, size=1).reason, "open_position_limit")

    def test_rejects_stale_signal(self):
        self.assertEqual(decide({}, size=1, created=NOW - 121).reason, "stale_signal")

    def test_accepts_trade_inside_all_limits(self):
        decision = decide({"positions": []}, size=5)
        self.assertTrue(decision.allowed)
        self.assertIsNone(decision.reason)


if __name__ == "__main__":
    unittest.main()
