"""Pure paper-trading price and state transitions."""

from __future__ import annotations

import json
from dataclasses import dataclass

from executable_quotes import fetch_executable_quote
from trading_risk import fee_adjusted_ev, market_fee_rate


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float


def _outcome_prices(market: dict) -> list[float]:
    raw_prices = market.get("outcomePrices", "[]")
    if isinstance(raw_prices, str):
        return [float(price) for price in json.loads(raw_prices)]
    return [float(price) for price in raw_prices]


def market_quote(market: dict, side: str = "YES") -> Quote:
    prices = _outcome_prices(market)
    if not prices:
        raise ValueError("Market has no YES outcome price")
    yes_price = float(prices[0])
    yes_bid = float(market.get("bestBid", yes_price))
    yes_ask = float(market.get("bestAsk", yes_price))
    if side.upper() == "NO":
        if "noBestBid" not in market or "noBestAsk" not in market:
            raise ValueError("Market has no independently executable NO quote")
        return Quote(
            bid=float(market["noBestBid"]),
            ask=float(market["noBestAsk"]),
        )
    return Quote(bid=yes_bid, ask=yes_ask)


def yes_quote(market: dict) -> Quote:
    return market_quote(market, "YES")


def no_quote(market: dict) -> Quote:
    return market_quote(market, "NO")


def revalidate_signal(
    signal: dict,
    market: dict,
    *,
    min_ev: float,
    max_price: float,
    max_spread: float,
) -> dict | None:
    side = str(signal.get("side") or signal.get("outcome_side") or "YES").upper()
    if side == "NO":
        no_quote = fetch_executable_quote(str(signal.get("token_id", "")))
        if no_quote is None:
            return None
        quote = Quote(no_quote.bid, no_quote.ask)
    else:
        quote = market_quote(market, side)
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
    if signal.get("raw_p") is not None:
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
