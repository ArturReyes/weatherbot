from __future__ import annotations

import unittest

from validation import PromotionPolicy, chronological_out_of_sample_report


def market(*, closed_at: str, pnl: float, probability: float, outcome: str, strategy: str = "calibrated_mean") -> dict:
    return {
        "status": "resolved",
        "resolved_outcome": outcome,
        "position": {
            "status": "closed",
            "closed_at": closed_at,
            "pnl": pnl,
            "cost": 1.0,
            "p": probability,
            "ev": 0.10,
            "entry_price": 0.50,
            "exit_price": 1.0 if outcome == "win" else 0.0,
            "close_reason": "resolved",
            "strategy": strategy,
        },
    }


class ChronologicalValidationTests(unittest.TestCase):
    def test_latest_records_are_the_only_holdout_records(self) -> None:
        markets = [
            market(closed_at=f"2026-07-0{day}T00:00:00+00:00", pnl=1.0, probability=0.8, outcome="win")
            for day in range(1, 9)
        ]
        report = chronological_out_of_sample_report(
            markets,
            policy=PromotionPolicy(
                holdout_fraction=0.25,
                min_holdout_trades=2,
                min_brier_samples=2,
                min_realized_roi=0.0,
                max_brier_score=0.25,
                max_drawdown_pct=0.10,
            ),
            bankroll=10.0,
        )

        self.assertEqual(report.training["calibrated_mean"].trades, 6)
        self.assertEqual(report.holdout["calibrated_mean"].trades, 2)
        self.assertAlmostEqual(report.holdout["calibrated_mean"].brier_score, 0.04)
        self.assertTrue(report.decision.ready)

    def test_holdout_gate_rejects_negative_realised_performance(self) -> None:
        markets = [
            market(closed_at="2026-07-01T00:00:00+00:00", pnl=1.0, probability=0.8, outcome="win"),
            market(closed_at="2026-07-02T00:00:00+00:00", pnl=-1.0, probability=0.8, outcome="loss"),
        ]
        report = chronological_out_of_sample_report(
            markets,
            policy=PromotionPolicy(
                holdout_fraction=0.5,
                min_holdout_trades=1,
                min_brier_samples=1,
                min_realized_roi=0.0,
                max_brier_score=0.25,
                max_drawdown_pct=0.50,
            ),
            bankroll=10.0,
        )

        self.assertFalse(report.decision.ready)
        self.assertTrue(any("realised ROI" in reason for reason in report.decision.reasons))
        self.assertTrue(any("Brier" in reason for reason in report.decision.reasons))

    def test_unresolved_early_exit_is_excluded_from_brier_scoring(self) -> None:
        early_exit = market(
            closed_at="2026-07-01T00:00:00+00:00",
            pnl=0.10,
            probability=0.70,
            outcome="win",
        )
        early_exit["status"] = "open"
        early_exit.pop("resolved_outcome")
        early_exit["position"]["close_reason"] = "take_profit"
        report = chronological_out_of_sample_report(
            [early_exit],
            policy=PromotionPolicy(min_holdout_trades=1, min_brier_samples=1),
            bankroll=10.0,
        )

        metrics = report.holdout["calibrated_mean"]
        self.assertEqual(metrics.brier_samples, 0)
        self.assertIsNone(metrics.brier_score)

    def test_required_enabled_strategy_cannot_bypass_promotion_without_samples(self) -> None:
        report = chronological_out_of_sample_report(
            [market(closed_at="2026-07-01T00:00:00+00:00", pnl=1.0, probability=0.8, outcome="win")],
            policy=PromotionPolicy(min_holdout_trades=1, min_brier_samples=1),
            bankroll=10.0,
            required_strategies=("calibrated_mean", "model_lag"),
        )

        self.assertFalse(report.decision.ready)
        self.assertIn("model_lag: no holdout trades", report.decision.reasons)
