"""Independent executable CLOB quotes for individual outcome tokens."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import requests


CLOB_BOOK_URL = "https://clob.polymarket.com/book"


@dataclass(frozen=True)
class ExecutableQuote:
    bid: float
    ask: float
    min_order_size: float
    tick_size: float


def fetch_executable_quote(
    token_id: str,
    *,
    get: Callable = requests.get,
) -> ExecutableQuote | None:
    """Read the best bid/ask from the token's public CLOB order book.

    No synthetic complement is permitted: missing bids or asks means the token
    is not executable for entry under this strategy.
    """
    if not token_id:
        return None
    try:
        response = get(CLOB_BOOK_URL, params={"token_id": token_id}, timeout=(3, 5))
        data = response.json()
        bids = [float(level["price"]) for level in data.get("bids", []) if level.get("price") is not None]
        asks = [float(level["price"]) for level in data.get("asks", []) if level.get("price") is not None]
        min_order_size = float(data["min_order_size"])
        tick_size = float(data["tick_size"])
    except (AttributeError, KeyError, TypeError, ValueError, requests.RequestException):
        return None
    if not bids or not asks or min_order_size <= 0 or tick_size <= 0:
        return None
    return ExecutableQuote(
        bid=max(bids),
        ask=min(asks),
        min_order_size=min_order_size,
        tick_size=tick_size,
    )
