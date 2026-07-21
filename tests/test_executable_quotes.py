from __future__ import annotations

import unittest

from executable_quotes import fetch_executable_quote


class _Response:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload


class ExecutableQuoteTests(unittest.TestCase):
    def test_uses_best_levels_from_the_token_order_book(self) -> None:
        calls: list[tuple[str, dict, tuple[int, int]]] = []

        def get(url: str, *, params: dict, timeout: tuple[int, int]) -> _Response:
            calls.append((url, params, timeout))
            return _Response({
                "bids": [{"price": "0.40"}, {"price": "0.42"}],
                "asks": [{"price": "0.47"}, {"price": "0.45"}],
                "min_order_size": "5",
                "tick_size": "0.01",
            })

        quote = fetch_executable_quote("no-token", get=get)

        self.assertEqual((quote.bid, quote.ask), (0.42, 0.45))
        self.assertEqual((quote.min_order_size, quote.tick_size), (5.0, 0.01))
        self.assertEqual(calls[0][1], {"token_id": "no-token"})

    def test_rejects_one_sided_or_empty_books(self) -> None:
        self.assertIsNone(fetch_executable_quote(
            "no-token",
            get=lambda *_args, **_kwargs: _Response({"bids": [{"price": "0.42"}], "asks": []}),
        ))

    def test_fails_closed_without_exchange_minimum_metadata(self) -> None:
        self.assertIsNone(fetch_executable_quote(
            "token",
            get=lambda *_args, **_kwargs: _Response({
                "bids": [{"price": "0.42"}],
                "asks": [{"price": "0.45"}],
            }),
        ))
