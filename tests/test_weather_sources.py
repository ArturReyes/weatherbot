from __future__ import annotations

import unittest
from unittest.mock import patch

import weatherbet


class _Response:
    def json(self) -> dict:
        return {
            "daily": {
                "time": ["2026-07-10"],
                "temperature_2m_max": [81.2],
            }
        }


class WeatherSourceTests(unittest.TestCase):
    def test_hrrr_uses_explicit_hrrr_conus_model(self) -> None:
        with patch("weatherbet.requests.get", return_value=_Response()) as request:
            result = weatherbet.get_hrrr("nyc", {"2026-07-10"})

        url = request.call_args.args[0]
        self.assertIn("/v1/gfs?", url)
        self.assertIn("models=ncep_hrrr_conus", url)
        self.assertEqual(result, {"2026-07-10": 81})

    def test_hrrr_is_not_requested_for_non_us_city(self) -> None:
        with patch("weatherbet.requests.get") as request:
            result = weatherbet.get_hrrr("london", {"2026-07-10"})

        self.assertEqual(result, {})
        request.assert_not_called()

    def test_snapshot_uses_explicit_source_name_for_real_hrrr(self) -> None:
        with (
            patch("weatherbet.get_ecmwf", return_value={"2026-07-10": 80}),
            patch("weatherbet.get_hrrr", return_value={"2026-07-10": 81}),
            patch("weatherbet.get_metar", return_value=None),
        ):
            snapshots = weatherbet.take_forecast_snapshot("nyc", ["2026-07-10"])

        snapshot = snapshots["2026-07-10"]
        self.assertEqual(snapshot["hrrr_conus"], 81)
        self.assertEqual(snapshot["best_source"], "hrrr_conus")
        self.assertNotIn("hrrr", snapshot)
