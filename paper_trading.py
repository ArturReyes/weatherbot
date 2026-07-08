"""Pure paper-trading price and state transitions."""

from __future__ import annotations

import json
from dataclasses import dataclass

from trading_risk import fee_adjusted_ev, market_fee_rate


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float


def yes_quote(market: dict) -> Quote:
    raw_prices = market.get("outcomePrices", "[]")
    if isinstance(raw_prices, str):
        prices = json.loads(raw_prices)
    else:
        prices = raw_prices
    if not prices:
        raise ValueError("Market has no YES outcome price")
    yes_price = float(prices[0])
    bid = float(market.get("bestBid", yes_price))
    ask = float(market.get("bestAsk", yes_price))
    return Quote(bid=bid, ask=ask)


def revalidate_signal(
    signal: dict,
    market: dict,
    *,
    min_ev: float,
    max_price: float,
    max_spread: float,
) -> dict | None:
    quote = yes_quote(market)
    spread = quote.ask - quote.bid
    if quote.ask >= max_price or spread > max_spread:
        return None
    fee_rate = market_fee_rate(market)
    ev = fee_adjusted_ev(
        probability=float(signal["p"]),
        price=quote.ask,
        fee_rate=fee_rate,
    )
    if ev < min_ev:
        return None
    fee_per_share = fee_rate * quote.ask * (1.0 - quote.ask)
    total_cost_per_share = quote.ask + fee_per_share
    updated = dict(signal)
    updated.update(
        {
            "entry_price": quote.ask,
            "bid_at_entry": quote.bid,
            "spread": round(spread, 4),
            "fee_rate": fee_rate,
            "ev": ev,
            "shares": round(float(signal["cost"]) / total_cost_per_share, 2),
        }
    )
    if "raw_p" in signal:
        updated["raw_ev"] = fee_adjusted_ev(
            probability=float(signal["raw_p"]),
            price=quote.ask,
            fee_rate=fee_rate,
        )
    return updated


def close_position(
    position: dict,
    *,
    balance: float,
    current_price: float,
    reason: str,
    closed_at: str,
) -> tuple[float, bool]:
    if position.get("status") != "open":
        return balance, False
    shares = float(position["shares"])
    fee_rate = float(position.get("fee_rate", 0))
    exit_fee = shares * fee_rate * current_price * (1.0 - current_price)
    proceeds = shares * current_price - exit_fee
    pnl = round(proceeds - float(position["cost"]), 2)
    position.update(
        {
            "closed_at": closed_at,
            "close_reason": reason,
            "exit_price": current_price,
            "exit_fee": round(exit_fee, 5),
            "pnl": pnl,
            "status": "closed",
        }
    )
    return round(balance + proceeds, 2), True
