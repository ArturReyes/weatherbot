from __future__ import annotations

import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from paper_trading import (
    archive_position_for_reentry,
    close_position,
    market_positions,
    no_quote,
    paper_reentry_reason,
    record_shadow_signal,
    revalidate_signal,
    settle_shadow_signals,
    settle_paper_market,
    yes_quote,
)


class PaperQuoteTests(unittest.TestCase):
    def test_yes_quote_uses_best_bid_and_ask_not_no_outcome_price(self) -> None:
        market = {
            "outcomePrices": '["0.20", "0.80"]',
            "bestBid": 0.19,
            "bestAsk": 0.21,
        }

        quote = yes_quote(market)

        self.assertEqual(quote.bid, 0.19)
        self.assertEqual(quote.ask, 0.21)

    def test_yes_quote_falls_back_to_yes_price_only(self) -> None:
        quote = yes_quote({"outcomePrices": '["0.20", "0.80"]'})

        self.assertEqual(quote.bid, 0.20)
        self.assertEqual(quote.ask, 0.20)

    def test_no_quote_requires_independent_executable_prices(self) -> None:
        with self.assertRaisesRegex(ValueError, "independently executable"):
            no_quote({"outcomePrices": '["0.20", "0.80"]', "bestBid": 0.19, "bestAsk": 0.21})

    def test_no_quote_uses_explicit_no_prices(self) -> None:
        quote = no_quote({"outcomePrices": '["0.20", "0.80"]', "noBestBid": 0.78, "noBestAsk": 0.80})

        self.assertEqual(quote.bid, 0.78)
        self.assertEqual(quote.ask, 0.80)


class PaperSignalTests(unittest.TestCase):
    def test_refresh_rejects_signal_when_ev_falls_below_threshold(self) -> None:
        signal = {"token_id": "yes-token", "p": 0.30, "cost": 5.0}
        detail = {
            "outcomePrices": '["0.25", "0.75"]',
            "bestBid": 0.24,
            "bestAsk": 0.29,
            "feesEnabled": True,
            "feeSchedule": {"rate": 0.05},
        }

        with patch("paper_trading.fetch_executable_quote") as fetch_quote:
            fetch_quote.return_value = type("Quote", (), {
                "bid": 0.24, "ask": 0.29, "min_order_size": 5.0, "tick_size": 0.01,
            })()
            refreshed = revalidate_signal(
                signal,
                detail,
                min_ev=0.10,
                max_price=0.45,
                max_spread=0.10,
            )

        self.assertIsNone(refreshed)

    def test_refresh_updates_price_shares_fee_and_ev_together(self) -> None:
        signal = {"token_id": "yes-token", "p": 0.50, "raw_p": 0.30, "raw_ev": -0.05, "cost": 5.0}
        detail = {
            "outcomePrices": '["0.20", "0.80"]',
            "bestBid": 0.19,
            "bestAsk": 0.20,
            "feesEnabled": True,
            "feeSchedule": {"rate": 0.05},
        }

        with patch("paper_trading.fetch_executable_quote") as fetch_quote:
            fetch_quote.return_value = type("Quote", (), {
                "bid": 0.19, "ask": 0.20, "min_order_size": 5.0, "tick_size": 0.01,
            })()
            refreshed = revalidate_signal(
                signal,
                detail,
                min_ev=0.10,
                max_price=0.45,
                max_spread=0.03,
            )

        self.assertIsNotNone(refreshed)
        assert refreshed is not None
        self.assertEqual(refreshed["entry_price"], 0.20)
        self.assertEqual(refreshed["fee_rate"], 0.05)
        self.assertAlmostEqual(refreshed["shares"], 24.0385, places=4)
        self.assertEqual(refreshed["min_order_size"], 5.0)
        self.assertEqual(refreshed["raw_ev"], 0.4423)

    def test_no_revalidation_uses_the_no_token_order_book(self) -> None:
        signal = {"side": "NO", "token_id": "no-token", "p": 0.80, "cost": 5.0}
        with patch("paper_trading.fetch_executable_quote") as fetch_quote:
            fetch_quote.return_value = type("Quote", (), {
                "bid": 0.18, "ask": 0.20, "min_order_size": 5.0, "tick_size": 0.01,
            })()
            refreshed = revalidate_signal(
                signal,
                {"outcomePrices": '["0.20", "0.80"]'},
                min_ev=0.10,
                max_price=0.90,
                max_spread=0.03,
            )

        self.assertIsNotNone(refreshed)
        assert refreshed is not None
        self.assertEqual(refreshed["entry_price"], 0.20)
        fetch_quote.assert_called_once_with("no-token")

    def test_shadow_signal_is_deduplicated_and_resolved_without_cash(self) -> None:
        market = {}
        signal = {
            "market_id": "bucket-1",
            "token_id": "token-1",
            "strategy": "calibrated_mean",
            "side": "YES",
            "p": 0.70,
            "entry_price": 0.20,
            "proposed_notional": 0.19,
            "required_notional": 1.0,
            "min_order_size": 5,
        }

        self.assertTrue(record_shadow_signal(
            market, signal, recorded_at="2026-07-17T12:00:00+00:00", skip_reason="below_order_minimum"
        ))
        self.assertFalse(record_shadow_signal(
            market, signal, recorded_at="2026-07-17T12:01:00+00:00", skip_reason="below_order_minimum"
        ))
        self.assertEqual(settle_shadow_signals(
            market, winning_market_id="bucket-1", resolved_at="2026-07-18T00:00:00+00:00"
        ), 1)
        self.assertEqual(market["shadow_signals"][0]["eventual_outcome"], "win")


class PaperCloseTests(unittest.TestCase):
    def test_closed_position_cannot_credit_balance_twice(self) -> None:
        position = {
            "status": "open",
            "entry_price": 0.20,
            "shares": 10.0,
            "cost": 2.0,
        }

        balance, first_closed = close_position(
            position,
            balance=10.0,
            current_price=0.30,
            reason="stop_loss",
            closed_at="2026-07-08T12:00:00+00:00",
        )
        balance, second_closed = close_position(
            position,
            balance=balance,
            current_price=0.30,
            reason="forecast_changed",
            closed_at="2026-07-08T12:00:01+00:00",
        )

        self.assertTrue(first_closed)
        self.assertFalse(second_closed)
        self.assertEqual(balance, 13.0)
        self.assertEqual(position["close_reason"], "stop_loss")

    def test_reentry_requires_cooldown_and_preserves_closed_position(self) -> None:
        market = {
            "position": {
                "status": "closed",
                "closed_at": "2026-07-15T10:00:00+00:00",
                "pnl": -0.25,
            }
        }

        blocked = paper_reentry_reason(
            market,
            now=datetime(2026, 7, 15, 10, 30, tzinfo=timezone.utc),
            enabled=True,
            cooldown_minutes=60,
            max_entries=2,
        )
        allowed = paper_reentry_reason(
            market,
            now=datetime(2026, 7, 15, 11, 1, tzinfo=timezone.utc),
            enabled=True,
            cooldown_minutes=60,
            max_entries=2,
        )

        self.assertEqual(blocked, "paper re-entry cooldown")
        self.assertIsNone(allowed)
        archive_position_for_reentry(market)
        market["position"] = {"status": "open", "cost": 1.0}
        self.assertEqual(len(market_positions(market)), 2)
        self.assertEqual(market["position_history"][0]["pnl"], -0.25)


class PaperResolutionTests(unittest.TestCase):
    def test_resolution_tracks_archived_and_current_reentry_without_double_pnl(self) -> None:
        market = {
            "status": "closed",
            "position_history": [
                {
                    "status": "closed",
                    "side": "YES",
                    "cost": 2.0,
                    "shares": 10.0,
                    "pnl": -0.5,
                }
            ],
            "position": {
                "status": "open",
                "side": "YES",
                "cost": 1.0,
                "shares": 5.0,
            },
        }

        transition = settle_paper_market(
            market,
            yes_won=True,
            balance=7.0,
            resolved_at="2026-07-15T23:00:00+00:00",
        )

        self.assertEqual(transition.balance, 12.0)
        self.assertEqual(transition.recorded_results, (False, True))
        self.assertEqual(market["position_history"][0]["pnl"], -0.5)
        self.assertEqual(market["position"]["pnl"], 4.0)
        self.assertEqual(market["pnl"], 3.5)

    def test_early_exit_records_eventual_outcome_without_changing_balance_or_pnl(self) -> None:
        market = {
            "status": "closed",
            "position": {
                "status": "closed",
                "side": "YES",
                "cost": 2.0,
                "shares": 10.0,
                "pnl": -0.50,
                "exit_price": 0.15,
                "close_reason": "stop_loss",
                "closed_at": "2026-07-08T12:00:00+00:00",
            },
        }

        transition = settle_paper_market(
            market,
            yes_won=True,
            balance=10.0,
            resolved_at="2026-07-09T23:00:00+00:00",
            actual_temp=80.0,
        )

        self.assertTrue(transition.newly_resolved)
        self.assertFalse(transition.position_was_open)
        self.assertEqual(transition.balance, 10.0)
        self.assertEqual(market["position"]["pnl"], -0.50)
        self.assertEqual(market["position"]["exit_price"], 0.15)
        self.assertEqual(market["position"]["eventual_outcome"], "win")
        self.assertEqual(market["resolved_outcome"], "win")
        self.assertEqual(market["actual_temp"], 80.0)

        repeated = settle_paper_market(
            market,
            yes_won=True,
            balance=10.0,
            resolved_at="2026-07-09T23:01:00+00:00",
        )
        self.assertFalse(repeated.newly_resolved)
        self.assertEqual(repeated.balance, 10.0)

    def test_open_position_is_paid_once_at_resolution(self) -> None:
        market = {
            "status": "closed",
            "position": {
                "status": "open",
                "outcome_side": "YES",
                "cost": 2.0,
                "shares": 8.0,
            },
        }

        transition = settle_paper_market(
            market,
            yes_won=True,
            balance=8.0,
            resolved_at="2026-07-09T23:00:00+00:00",
        )

        self.assertEqual(transition.balance, 16.0)
        self.assertEqual(market["position"]["pnl"], 6.0)
        self.assertEqual(market["position"]["status"], "closed")
        self.assertEqual(market["position"]["close_reason"], "resolved")


if __name__ == "__main__":
    unittest.main()
