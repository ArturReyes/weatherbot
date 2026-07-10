from __future__ import annotations

import unittest
from datetime import datetime, timezone

from observations import daily_observed_high


class DailyObservedHighTests(unittest.TestCase):
    def test_full_day_coverage_returns_high_and_is_complete(self) -> None:
        result = daily_observed_high(
            [
                {"ts": "2026-07-10T00:30:00+00:00", "value": 70},
                {"ts": "2026-07-10T01:30:00+00:00", "value": 74},
                {"ts": "2026-07-10T02:30:00+00:00", "value": 72},
            ],
            market_date="2026-07-10",
            timezone_name="UTC",
            now=datetime(2026, 7, 10, 3, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(result.high, 74.0)
        self.assertTrue(result.complete)

    def test_late_start_fails_closed_even_with_a_current_temperature(self) -> None:
        result = daily_observed_high(
            [{"ts": "2026-07-10T15:00:00+00:00", "value": 84}],
            market_date="2026-07-10",
            timezone_name="UTC",
            now=datetime(2026, 7, 10, 15, 30, tzinfo=timezone.utc),
        )

        self.assertEqual(result.high, 84.0)
        self.assertFalse(result.complete)

    def test_collection_gap_fails_closed(self) -> None:
        result = daily_observed_high(
            [
                {"ts": "2026-07-10T00:30:00+00:00", "value": 70},
                {"ts": "2026-07-10T04:00:00+00:00", "value": 80},
            ],
            market_date="2026-07-10",
            timezone_name="UTC",
            now=datetime(2026, 7, 10, 4, 15, tzinfo=timezone.utc),
        )

        self.assertFalse(result.complete)
