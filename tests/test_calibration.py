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
            "actual_temp": 72,
            "position": {"opened_at": "2026-07-01T12:00:00+00:00"},
            "forecast_snapshots": [
                {"ts": "2026-07-01T11:00:00+00:00", "ecmwf": 70},
                {"ts": "2026-07-01T13:00:00+00:00", "ecmwf": 72},
            ],
        }]
        self.assertEqual(calibration_errors(markets, city="nyc", source="ecmwf"), [-2.0])


class SigmaTests(unittest.TestCase):
    def test_sigma_is_root_mean_square_error_not_mae(self):
        self.assertAlmostEqual(rmse_sigma([1, 3]), math.sqrt(5))

    def test_sigma_has_positive_floor(self):
        self.assertEqual(rmse_sigma([0, 0], floor=0.5), 0.5)

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
            "actual_temp": 70,
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
                "resolved": True,
                "actual_temp": 70,
                "forecast_snapshots": [
                    {"ts": "2026-07-01T12:00:00+00:00", "ecmwf": 74},
                ],
            },
            {
                "city": "nyc",
                "resolved": True,
                "actual_temp": 71,
                "forecast_snapshots": [
                    {"ts": "2026-07-02T12:00:00+00:00", "ecmwf": 75},
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


class RunCalibrationTests(unittest.TestCase):
    def test_run_calibration_writes_bias_and_sigma_per_lead_bucket(self):
        markets = [
            {
                "city": "nyc",
                "resolved": True,
                "actual_temp": 70,
                "event_end_date": "2026-07-03T00:00:00+00:00",
                "position": {"opened_at": "2026-07-01T13:00:00+00:00"},
                "forecast_snapshots": [
                    {"ts": "2026-07-01T12:00:00+00:00", "hrrr_conus": 74},
                    {"ts": "2026-07-01T14:00:00+00:00", "hrrr_conus": 70},
                ],
            },
            {
                "city": "nyc",
                "resolved": True,
                "actual_temp": 71,
                "event_end_date": "2026-07-04T00:00:00+00:00",
                "position": {"opened_at": "2026-07-02T13:00:00+00:00"},
                "forecast_snapshots": [
                    {"ts": "2026-07-02T12:00:00+00:00", "hrrr_conus": 75},
                    {"ts": "2026-07-02T14:00:00+00:00", "hrrr_conus": 71},
                ],
            },
        ]

        original_file = weatherbet.CALIBRATION_FILE
        original_min = weatherbet.CALIBRATION_MIN
        original_decay = weatherbet.BIAS_DECAY
        original_prior = weatherbet.BIAS_PRIOR_STRENGTH
        original_cal = weatherbet._cal

        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                weatherbet.CALIBRATION_FILE = Path(temp_dir) / "calibration.json"
                weatherbet.CALIBRATION_MIN = 2
                weatherbet.BIAS_DECAY = 1.0
                weatherbet.BIAS_PRIOR_STRENGTH = 0.0
                weatherbet.install_calibration({})

                cal = weatherbet.run_calibration(markets)
            finally:
                weatherbet.CALIBRATION_FILE = original_file
                weatherbet.CALIBRATION_MIN = original_min
                weatherbet.BIAS_DECAY = original_decay
                weatherbet.BIAS_PRIOR_STRENGTH = original_prior
                weatherbet.install_calibration(original_cal)

        self.assertEqual(cal["nyc_hrrr_conus"]["sigma"], 4.0)
        self.assertEqual(cal["nyc_hrrr_conus"]["n"], 2)
        self.assertEqual(cal["nyc_hrrr_conus_24_48h"]["bias"], 4.0)
        self.assertEqual(cal["nyc_hrrr_conus_24_48h"]["raw_bias"], 4.0)
        self.assertEqual(cal["nyc_hrrr_conus_24_48h"]["sigma"], 4.0)
        self.assertEqual(cal["nyc_hrrr_conus_24_48h"]["n"], 2)
        self.assertEqual(cal["nyc_hrrr_conus_24_48h"]["lead_bucket"], "24_48h")


if __name__ == "__main__":
    unittest.main()
