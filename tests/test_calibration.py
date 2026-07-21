import math
import tempfile
import unittest
from pathlib import Path

import weatherbet
from calibration import (
    bias_adjusted_forecast,
    calibration_errors,
    decaying_mean_error,
    lead_time_bucket,
    regularized_sigma,
    rmse_sigma,
    select_calibration_snapshot,
)


class SnapshotSelectionTests(unittest.TestCase):
    def test_snapshot_after_position_entry_is_ignored(self):
        market = {
            "position": {"opened_at": "2026-07-01T12:00:00+00:00"},
            "forecast_snapshots": [
                {"ts": "2026-07-01T11:00:00+00:00", "ecmwf": 70},
                {"ts": "2026-07-01T13:00:00+00:00", "ecmwf": 80},
            ],
        }
        self.assertEqual(select_calibration_snapshot(market, "ecmwf")["ecmwf"], 70)

    def test_market_without_position_uses_earliest_snapshot(self):
        market = {
            "forecast_snapshots": [
                {"ts": "2026-07-01T13:00:00+00:00", "hrrr": 80},
                {"ts": "2026-07-01T11:00:00+00:00", "hrrr": 70},
            ]
        }
        self.assertEqual(select_calibration_snapshot(market, "hrrr")["hrrr"], 70)

    def test_error_uses_only_selected_non_lookahead_snapshot(self):
        markets = [{
            "city": "nyc",
            "resolved": True,
            "calibration_temp": 72,
            "calibration_source": "polymarket_winning_bucket",
            "position": {"opened_at": "2026-07-01T12:00:00+00:00"},
            "forecast_snapshots": [
                {"ts": "2026-07-01T11:00:00+00:00", "ecmwf": 70},
                {"ts": "2026-07-01T13:00:00+00:00", "ecmwf": 72},
            ],
        }]
        self.assertEqual(calibration_errors(markets, city="nyc", source="ecmwf"), [-2.0])

    def test_closed_market_with_only_provider_actual_is_not_calibration_eligible(self):
        markets = [{
            "city": "nyc",
            "status": "closed",
            "actual_temp": 72,
            "forecast_snapshots": [
                {"ts": "2026-07-01T11:00:00+00:00", "ecmwf": 74},
            ],
        }]

        self.assertEqual(calibration_errors(markets, city="nyc", source="ecmwf"), [])

    def test_closed_market_with_polymarket_calibration_temp_is_eligible(self):
        markets = [{
            "city": "nyc",
            "status": "closed",
            "actual_temp": 90,
            "calibration_temp": 72,
            "calibration_source": "polymarket_winning_bucket",
            "forecast_snapshots": [
                {"ts": "2026-07-01T11:00:00+00:00", "ecmwf": 74},
            ],
        }]

        self.assertEqual(calibration_errors(markets, city="nyc", source="ecmwf"), [2.0])

    def test_lead_bucket_selects_time_safe_snapshot_from_that_bucket(self):
        market = {
            "event_end_date": "2026-07-03T00:00:00+00:00",
            "forecast_snapshots": [
                {"ts": "2026-06-30T12:00:00+00:00", "ecmwf": 76},
                {"ts": "2026-07-01T12:00:00+00:00", "ecmwf": 74},
                {"ts": "2026-07-02T12:00:00+00:00", "ecmwf": 72},
            ],
        }

        selected = select_calibration_snapshot(
            market,
            "ecmwf",
            lead_bucket="24_48h",
        )

        self.assertEqual(selected["ecmwf"], 74)


class SigmaTests(unittest.TestCase):
    def test_sigma_is_root_mean_square_error_not_mae(self):
        self.assertAlmostEqual(rmse_sigma([1, 3]), math.sqrt(5))

    def test_sigma_has_positive_floor(self):
        self.assertEqual(rmse_sigma([0, 0], floor=0.5), 0.5)

    def test_regularized_sigma_shrinks_small_sample_variance_to_prior(self):
        sigma = regularized_sigma(
            [4.0] * 7,
            prior_sigma=2.0,
            prior_strength=21.0,
        )

        self.assertAlmostEqual(sigma, math.sqrt(7.0))

    def test_loaded_calibration_is_used_by_get_sigma(self):
        original = weatherbet._cal
        try:
            weatherbet.install_calibration({"nyc_ecmwf": {"sigma": 3.25}})
            self.assertEqual(weatherbet.get_sigma("nyc", "ecmwf"), 3.25)
        finally:
            weatherbet.install_calibration(original)


class BiasCorrectionTests(unittest.TestCase):
    def test_lead_time_bucket_uses_snapshot_to_event_end(self):
        self.assertEqual(
            lead_time_bucket(
                "2026-07-01T12:00:00+00:00",
                "2026-07-03T00:00:00+00:00",
            ),
            "24_48h",
        )
        self.assertEqual(
            lead_time_bucket(
                "2026-07-01T12:00:00+00:00",
                "2026-07-05T00:00:00+00:00",
            ),
            "72h_plus",
        )
        self.assertIsNone(
            lead_time_bucket(
                "2026-07-03T00:00:00+00:00",
                "2026-07-01T12:00:00+00:00",
            )
        )

    def test_decaying_mean_error_ignores_post_entry_snapshot(self):
        markets = [{
            "city": "nyc",
            "resolved": True,
            "calibration_temp": 70,
            "calibration_source": "polymarket_winning_bucket",
            "event_end_date": "2026-07-03T00:00:00+00:00",
            "position": {"opened_at": "2026-07-01T13:00:00+00:00"},
            "forecast_snapshots": [
                {"ts": "2026-07-01T12:00:00+00:00", "hrrr": 74},
                {"ts": "2026-07-01T14:00:00+00:00", "hrrr": 70},
            ],
        }]

        estimate = decaying_mean_error(
            markets,
            city="nyc",
            source="hrrr",
            lead_bucket="24_48h",
            decay=1.0,
            prior_strength=0.0,
        )

        self.assertEqual(estimate.n, 1)
        self.assertEqual(estimate.raw_bias, 4.0)
        self.assertEqual(estimate.bias, 4.0)

    def test_decaying_mean_error_shrinks_small_sample_bias(self):
        markets = [
            {
                "city": "nyc",
                "station": "KLGA",
                "resolved": True,
                "calibration_temp": 70,
                "calibration_source": "polymarket_winning_bucket",
                "forecast_snapshots": [
                    {"ts": "2026-07-01T12:00:00+00:00", "ecmwf": 74},
                ],
            },
            {
                "city": "nyc",
                "station": "KLGA",
                "resolved": True,
                "calibration_temp": 71,
                "calibration_source": "polymarket_winning_bucket",
                "forecast_snapshots": [
                    {"ts": "2026-07-02T12:00:00+00:00", "ecmwf": 75},
                ],
            },
            {
                "city": "nyc",
                "station": "KJFK",
                "resolved": True,
                "calibration_temp": 0,
                "calibration_source": "polymarket_winning_bucket",
                "event_end_date": "2026-07-05T00:00:00+00:00",
                "forecast_snapshots": [
                    {"ts": "2026-07-03T12:00:00+00:00", "hrrr_conus": 100},
                ],
            },
        ]

        estimate = decaying_mean_error(
            markets,
            city="nyc",
            source="ecmwf",
            decay=1.0,
            prior_strength=2.0,
        )

        self.assertEqual(estimate.n, 2)
        self.assertEqual(estimate.raw_bias, 4.0)
        self.assertEqual(estimate.bias, 2.0)

    def test_bias_adjusted_forecast_subtracts_and_caps_bias(self):
        self.assertEqual(bias_adjusted_forecast(80, 2), 78)
        self.assertEqual(bias_adjusted_forecast(80, -2), 82)
        self.assertEqual(
            bias_adjusted_forecast(80, 5, max_correction=3),
            77,
        )

    def test_forecast_calibration_returns_capped_corrected_metadata(self):
        original = weatherbet._cal
        try:
            weatherbet.install_calibration({
                "nyc_ecmwf_24_48h": {
                    "bias": 5.0,
                    "raw_bias": 5.5,
                    "sigma": 1.75,
                    "n": 9,
                }
            })

            meta = weatherbet.forecast_calibration(
                "nyc",
                "ecmwf",
                82,
                "2026-07-01T12:00:00+00:00",
                "2026-07-03T00:00:00+00:00",
            )
        finally:
            weatherbet.install_calibration(original)

        self.assertEqual(meta["raw_forecast_temp"], 82.0)
        self.assertEqual(meta["corrected_forecast_temp"], 79.0)
        self.assertEqual(meta["forecast_bias"], 3.0)
        self.assertEqual(meta["forecast_raw_bias"], 5.5)
        self.assertEqual(meta["forecast_lead_bucket"], "24_48h")
        self.assertEqual(meta["forecast_calibration_n"], 9)
        self.assertEqual(meta["sigma"], 1.75)

    def test_forecast_calibration_falls_back_to_aggregate_bootstrap(self):
        original = weatherbet._cal
        try:
            weatherbet.install_calibration({
                "nyc_ecmwf": {
                    "bias": 1.0,
                    "raw_bias": 2.0,
                    "sigma": 2.5,
                    "n": 7,
                }
            })

            meta = weatherbet.forecast_calibration(
                "nyc",
                "ecmwf",
                82,
                "2026-07-01T12:00:00+00:00",
                "2026-07-03T00:00:00+00:00",
            )
        finally:
            weatherbet.install_calibration(original)

        self.assertEqual(meta["corrected_forecast_temp"], 81.0)
        self.assertEqual(meta["forecast_calibration_n"], 7)
        self.assertEqual(meta["forecast_calibration_scope"], "aggregate")
        self.assertEqual(meta["sigma"], 2.5)


class RunCalibrationTests(unittest.TestCase):
    def test_station_change_resets_only_unpositioned_market_observations(self):
        market = {
            "station": "LFPG",
            "position": None,
            "forecast_snapshots": [{"best": 25}],
            "market_snapshots": [{"top_price": 0.5}],
            "all_outcomes": [{"market_id": "old"}],
            "calibration_temp": 24,
        }

        aligned, message = weatherbet.align_unpositioned_market_station(
            market,
            weatherbet.LOCATIONS["paris"],
            changed_at="2026-07-21T12:00:00+00:00",
        )

        self.assertTrue(aligned, message)
        self.assertEqual(market["station"], "LFPB")
        self.assertEqual(market["forecast_snapshots"], [])
        self.assertEqual(market["all_outcomes"], [])
        self.assertNotIn("calibration_temp", market)

    def test_run_calibration_writes_bias_and_sigma_per_lead_bucket(self):
        markets = [
            {
                "city": "nyc",
                "station": "KLGA",
                "resolved": True,
                "calibration_temp": 70,
                "calibration_source": "polymarket_winning_bucket",
                "event_end_date": "2026-07-03T00:00:00+00:00",
                "position": {"opened_at": "2026-07-01T13:00:00+00:00"},
                "forecast_snapshots": [
                    {"ts": "2026-07-01T12:00:00+00:00", "hrrr_conus": 74},
                    {"ts": "2026-07-01T14:00:00+00:00", "hrrr_conus": 70},
                ],
            },
            {
                "city": "nyc",
                "station": "KLGA",
                "resolved": True,
                "calibration_temp": 71,
                "calibration_source": "polymarket_winning_bucket",
                "event_end_date": "2026-07-04T00:00:00+00:00",
                "position": {"opened_at": "2026-07-02T13:00:00+00:00"},
                "forecast_snapshots": [
                    {"ts": "2026-07-02T12:00:00+00:00", "hrrr_conus": 75},
                    {"ts": "2026-07-02T14:00:00+00:00", "hrrr_conus": 71},
                ],
            },
            {
                "city": "nyc",
                "station": "KJFK",
                "resolved": True,
                "calibration_temp": 0,
                "calibration_source": "polymarket_winning_bucket",
                "event_end_date": "2026-07-05T00:00:00+00:00",
                "forecast_snapshots": [
                    {"ts": "2026-07-03T12:00:00+00:00", "hrrr_conus": 100},
                ],
            },
        ]

        original_file = weatherbet.CALIBRATION_FILE
        original_min = weatherbet.CALIBRATION_MIN
        original_bootstrap_min = weatherbet.CALIBRATION_BOOTSTRAP_MIN
        original_decay = weatherbet.BIAS_DECAY
        original_prior = weatherbet.BIAS_PRIOR_STRENGTH
        original_cal = weatherbet._cal

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                weatherbet.CALIBRATION_FILE = Path(temp_dir) / "calibration.json"
                weatherbet.CALIBRATION_MIN = 2
                weatherbet.CALIBRATION_BOOTSTRAP_MIN = 2
                weatherbet.BIAS_DECAY = 1.0
                weatherbet.BIAS_PRIOR_STRENGTH = 0.0
                weatherbet.install_calibration({})

                cal = weatherbet.run_calibration(markets)
            finally:
                weatherbet.CALIBRATION_FILE = original_file
                weatherbet.CALIBRATION_MIN = original_min
                weatherbet.CALIBRATION_BOOTSTRAP_MIN = original_bootstrap_min
                weatherbet.BIAS_DECAY = original_decay
                weatherbet.BIAS_PRIOR_STRENGTH = original_prior
                weatherbet.install_calibration(original_cal)

        self.assertEqual(cal["nyc_hrrr_conus"]["sigma"], 4.0)
        self.assertEqual(cal["nyc_hrrr_conus"]["n"], 2)
        self.assertEqual(cal["nyc_hrrr_conus"]["station"], "KLGA")
        self.assertEqual(cal["nyc_hrrr_conus_24_48h"]["bias"], 4.0)
        self.assertEqual(cal["nyc_hrrr_conus_24_48h"]["raw_bias"], 4.0)
        self.assertEqual(cal["nyc_hrrr_conus_24_48h"]["sigma"], 4.0)
        self.assertEqual(cal["nyc_hrrr_conus_24_48h"]["n"], 2)
        self.assertEqual(cal["nyc_hrrr_conus_24_48h"]["lead_bucket"], "24_48h")


if __name__ == "__main__":
    unittest.main()
