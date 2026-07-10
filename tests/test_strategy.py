from __future__ import annotations

import unittest

from strategy import (
    EXIT_HOLD_TO_RESOLUTION,
    NEAR_LOCK,
    NO,
    UNDERDISPERSION_TAIL,
    YES,
    BucketQuote,
    ForecastContext,
    StrategyConfig,
    dispersion_ratio,
    generate_strategy_candidates,
    near_lock_probability,
)


def bucket(
    low: float,
    high: float,
    *,
    yes_ask: float = 0.10,
    yes_bid: float = 0.09,
    no_ask: float = 0.91,
    no_bid: float = 0.90,
    no_quote_verified: bool = False,
) -> BucketQuote:
    return BucketQuote(
        market_id=f"{low}-{high}",
        question="bucket",
        bucket_low=low,
        bucket_high=high,
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        volume=1000,
        fee_rate=0.0,
        yes_token_id="yes-token",
        no_token_id="no-token",
        no_quote_verified=no_quote_verified,
    )


def context(**overrides) -> ForecastContext:
    values = {
        "city_slug": "nyc",
        "unit": "F",
        "hours_left": 36.0,
        "horizon": "D+2",
        "raw_forecast_temp": 80.0,
        "corrected_forecast_temp": 80.0,
        "forecast_source": "ecmwf",
        "sigma": 4.0,
    }
    values.update(overrides)
    return ForecastContext(**values)


class StrategyTests(unittest.TestCase):
    def test_near_lock_probability_from_observed_high_and_remaining_forecast(self) -> None:
        prob = near_lock_probability(
            observed_high=84.2,
            forecast_remaining_max=84.3,
            low=84,
            high=85,
            unit="F",
            sigma=0.2,
        )

        self.assertGreater(prob, 0.92)

    def test_underdispersion_ratio_compares_historical_sigma_to_ensemble_spread(self) -> None:
        self.assertEqual(dispersion_ratio(historical_sigma=6.4, ensemble_spread=4.0), 1.6)

    def test_all_bucket_scanning_finds_tail_when_underdispersed(self) -> None:
        candidates = generate_strategy_candidates(
            buckets=[bucket(80, 80, yes_ask=0.30), bucket(87, 87, yes_ask=0.01, yes_bid=0.009)],
            context=context(ensemble_spread=2.0),
            config=StrategyConfig(
                enable_no_trades=False,
                strategy_underdispersion_enabled=True,
            ),
        )

        self.assertTrue(any(c.strategy == UNDERDISPERSION_TAIL and c.bucket_low == 87 for c in candidates))

    def test_source_disagreement_cannot_trigger_underdispersion(self) -> None:
        candidates = generate_strategy_candidates(
            buckets=[bucket(87, 87, yes_ask=0.01, yes_bid=0.009)],
            context=context(source_spread=2.0, ensemble_spread=None),
            config=StrategyConfig(
                enable_no_trades=False,
                strategy_calibrated_mean_enabled=False,
                strategy_underdispersion_enabled=True,
            ),
        )

        self.assertEqual(candidates, [])

    def test_model_lag_requires_insufficient_yes_repricing(self) -> None:
        candidates = generate_strategy_candidates(
            buckets=[bucket(80, 80, yes_ask=0.20, yes_bid=0.19, no_ask=0.81, no_bid=0.80)],
            context=context(
                corrected_forecast_temp=80,
                previous_corrected_forecast_temp=74,
                sigma=2.0,
            ),
            config=StrategyConfig(
                enable_no_trades=False,
                strategy_calibrated_mean_enabled=False,
                strategy_underdispersion_enabled=False,
            ),
        )

        self.assertEqual(candidates, [])

        lagging_bucket = bucket(80, 80, yes_ask=0.10, yes_bid=0.09)
        lagging_bucket = BucketQuote(
            **{**lagging_bucket.__dict__, "previous_yes_ask": 0.05}
        )
        candidates = generate_strategy_candidates(
            buckets=[lagging_bucket],
            context=context(
                corrected_forecast_temp=80,
                previous_corrected_forecast_temp=74,
                sigma=2.0,
            ),
            config=StrategyConfig(
                enable_no_trades=False,
                min_ev=0.0,
                strategy_calibrated_mean_enabled=False,
                strategy_underdispersion_enabled=False,
            ),
        )

        self.assertTrue(any(candidate.strategy == "model_lag" for candidate in candidates))

    def test_model_lag_rejects_market_that_already_repriced(self) -> None:
        repriced_bucket = bucket(80, 80, yes_ask=0.18, yes_bid=0.17)
        repriced_bucket = BucketQuote(
            **{**repriced_bucket.__dict__, "previous_yes_ask": 0.05}
        )
        candidates = generate_strategy_candidates(
            buckets=[repriced_bucket],
            context=context(
                corrected_forecast_temp=80,
                previous_corrected_forecast_temp=74,
                sigma=2.0,
            ),
            config=StrategyConfig(
                enable_no_trades=False,
                min_ev=0.0,
                strategy_calibrated_mean_enabled=False,
                strategy_underdispersion_enabled=False,
            ),
        )

        self.assertEqual(candidates, [])

    def test_yes_and_no_candidates_use_opposite_probabilities(self) -> None:
        candidates = generate_strategy_candidates(
            buckets=[bucket(90, 90, yes_ask=0.80, yes_bid=0.79, no_ask=0.21, no_bid=0.20, no_quote_verified=True)],
            context=context(corrected_forecast_temp=80, sigma=2.0),
            config=StrategyConfig(enable_no_trades=True, no_trade_min_ev=0.15),
        )

        no_candidates = [c for c in candidates if c.side == NO]
        self.assertTrue(no_candidates)
        self.assertGreater(no_candidates[0].probability, 0.95)
        self.assertEqual(no_candidates[0].token_id, "no-token")

    def test_strategy_specific_price_cap_allows_near_lock_high_price(self) -> None:
        candidates = generate_strategy_candidates(
            buckets=[bucket(84, 85, yes_ask=0.80, yes_bid=0.79)],
            context=context(
                hours_left=4.0,
                observed_high_so_far=84.4,
                observed_high_complete=True,
                forecast_remaining_max=84.3,
            ),
            config=StrategyConfig(enable_no_trades=False, near_lock_sigma_f=0.2),
        )

        self.assertTrue(candidates)
        self.assertEqual(candidates[0].strategy, NEAR_LOCK)
        self.assertEqual(candidates[0].exit_policy, EXIT_HOLD_TO_RESOLUTION)
        self.assertEqual(candidates[0].side, YES)

    def test_near_lock_fails_closed_without_complete_station_history(self) -> None:
        candidates = generate_strategy_candidates(
            buckets=[bucket(84, 85, yes_ask=0.80, yes_bid=0.79)],
            context=context(
                hours_left=4.0,
                observed_high_so_far=84.4,
                observed_high_complete=False,
                forecast_remaining_max=84.3,
            ),
            config=StrategyConfig(
                enable_no_trades=False,
                strategy_calibrated_mean_enabled=False,
            ),
        )

        self.assertEqual(candidates, [])


if __name__ == "__main__":
    unittest.main()
