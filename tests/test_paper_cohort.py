from __future__ import annotations

import unittest

from paper_cohort import (
    build_cohort_archive,
    build_fresh_paper_state,
    cohort_position_records,
    ensure_paper_reset_allowed,
    mark_market_positions_legacy,
)


class PaperCohortTests(unittest.TestCase):
    def test_reset_fails_closed_with_open_exposure(self) -> None:
        markets = [{"position": {"status": "open"}}]

        with self.assertRaisesRegex(ValueError, "open position"):
            ensure_paper_reset_allowed(markets, {})

    def test_archive_preserves_trade_records_and_fresh_state_resets_only_counters(self) -> None:
        market = {
            "city": "paris",
            "city_name": "Paris",
            "date": "2026-07-14",
            "status": "resolved",
            "actual_temp": 32.0,
            "position": {
                "status": "closed",
                "opened_at": "2026-07-14T10:00:00+00:00",
                "pnl": -1.25,
            },
        }
        state = {
            "balance": 7.68,
            "starting_balance": 25.0,
            "total_trades": 36,
            "wins": 1,
            "losses": 30,
            "peak_balance": 25.0,
        }

        archive = build_cohort_archive(
            markets=[market],
            state=state,
            cohort_id="legacy",
            ended_at="2026-07-15T16:00:00Z",
            evaluation_started_at="2026-07-10T00:00:00Z",
        )
        fresh = build_fresh_paper_state(
            previous_state=state,
            bankroll=25.0,
            cohort_id="paper-new",
            started_at="2026-07-15T16:00:00Z",
            archive_path="data/evaluations/archive.json",
        )

        self.assertEqual(archive["summary"]["positions"], 1)
        self.assertEqual(archive["summary"]["realized_pnl"], -1.25)
        self.assertEqual(archive["trades"][0]["actual_temp"], 32.0)
        self.assertEqual(fresh["balance"], 25.0)
        self.assertEqual(fresh["total_trades"], 0)
        self.assertEqual(fresh["paper_cohort_archives"], ["data/evaluations/archive.json"])

    def test_legacy_mark_prevents_late_resolution_from_polluting_fresh_counters(self) -> None:
        market = {
            "position": {
                "status": "closed",
                "pnl": -0.5,
                "trade_result_recorded": False,
            }
        }

        changed = mark_market_positions_legacy(market, cohort_id="legacy")

        self.assertTrue(changed)
        self.assertEqual(market["position"]["paper_cohort_id"], "legacy")
        self.assertTrue(market["position"]["trade_result_recorded"])

    def test_status_cohort_filter_uses_entry_timestamp(self) -> None:
        markets = [
            {
                "position_history": [
                    {
                        "status": "closed",
                        "opened_at": "2026-07-14T10:00:00Z",
                    }
                ],
                "position": {
                    "status": "open",
                    "opened_at": "2026-07-15T17:00:00Z",
                },
            }
        ]

        records = cohort_position_records(
            markets,
            started_at="2026-07-15T16:00:00Z",
        )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0][1]["status"], "open")


if __name__ == "__main__":
    unittest.main()
