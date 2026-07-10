from __future__ import annotations

import unittest
from unittest.mock import patch

from paper_trading import close_position, no_quote, revalidate_signal, yes_quote


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
        signal = {"p": 0.30, "cost": 5.0}
        detail = {
            "outcomePrices": '["0.25", "0.75"]',
            "bestBid": 0.24,
            "bestAsk": 0.29,
            "feesEnabled": True,
            "feeSchedule": {"rate": 0.05},
        }

        refreshed = revalidate_signal(
            signal,
            detail,
            min_ev=0.10,
            max_price=0.45,
            max_spread=0.10,
        )

        self.assertIsNone(refreshed)

    def test_refresh_updates_price_shares_fee_and_ev_together(self) -> None:
        signal = {"p": 0.50, "raw_p": 0.30, "raw_ev": -0.05, "cost": 5.0}
        detail = {
            "outcomePrices": '["0.20", "0.80"]',
            "bestBid": 0.19,
            "bestAsk": 0.20,
            "feesEnabled": True,
            "feeSchedule": {"rate": 0.05},
        }

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
        self.assertEqual(refreshed["shares"], 24.04)
        self.assertEqual(refreshed["raw_ev"], 0.4423)

    def test_no_revalidation_uses_the_no_token_order_book(self) -> None:
        signal = {"side": "NO", "token_id": "no-token", "p": 0.80, "cost": 5.0}
        with patch("paper_trading.fetch_executable_quote") as fetch_quote:
            fetch_quote.return_value = type("Quote", (), {"bid": 0.18, "ask": 0.20})()
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


if __name__ == "__main__":
    unittest.main()
