from __future__ import annotations

import unittest

from market_resolution import calibration_fields, winning_bucket_from_outcomes


class MarketResolutionTests(unittest.TestCase):
    def test_unique_bounded_winner_produces_midpoint_calibration(self) -> None:
        decision = winning_bucket_from_outcomes([
            {"market_id": "low", "range": [86, 87], "settlement_yes_price": 0.0},
            {"market_id": "winner", "range": [88, 89], "settlement_yes_price": 1.0},
        ])

        self.assertIsNotNone(decision.bucket)
        assert decision.bucket is not None
        self.assertEqual(decision.bucket.representative_temp, 88.5)
        fields = calibration_fields(
            decision.bucket,
            provider="wunderground",
            station="KLGA",
            validated_at="2026-07-17T12:00:00+00:00",
        )
        self.assertEqual(fields["calibration_temp"], 88.5)
        self.assertEqual(fields["calibration_source"], "polymarket_winning_bucket")

    def test_exact_bucket_overrides_conflicting_provider_temperature(self) -> None:
        market = {"actual_temp": 38.0}
        decision = winning_bucket_from_outcomes([
            {"market_id": "lucknow-33", "range": [33, 33], "price": 1.0},
        ], allow_legacy_price=True)

        assert decision.bucket is not None
        market.update(calibration_fields(
            decision.bucket,
            provider="polymarket_legacy",
            station="VILK",
            validated_at="2026-07-17T12:00:00+00:00",
        ))
        self.assertEqual(market["actual_temp"], 38.0)
        self.assertEqual(market["calibration_temp"], 33.0)

    def test_open_ended_winner_is_not_a_calibration_temperature(self) -> None:
        decision = winning_bucket_from_outcomes([
            {"market_id": "tail", "range": [35, 999], "settlement_yes_price": 1.0},
        ])

        assert decision.bucket is not None
        self.assertFalse(decision.bucket.bounded)
        with self.assertRaises(ValueError):
            calibration_fields(
                decision.bucket,
                provider="wunderground",
                station="VILK",
                validated_at="2026-07-17T12:00:00+00:00",
            )

    def test_ambiguous_winner_fails_closed(self) -> None:
        decision = winning_bucket_from_outcomes([
            {"market_id": "one", "range": [30, 30], "settlement_yes_price": 1.0},
            {"market_id": "two", "range": [31, 31], "settlement_yes_price": 1.0},
        ])

        self.assertIsNone(decision.bucket)
        self.assertEqual(decision.reason, "ambiguous_winner")
