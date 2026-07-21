"""Pure paper-trading price and state transitions."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from executable_quotes import fetch_executable_quote
from order_eligibility import OrderEligibility, evaluate_order_eligibility
from trading_risk import fee_adjusted_ev, market_fee_rate


@dataclass(frozen=True)
class Quote:
    bid: float
    ask: float


@dataclass(frozen=True)
class ResolutionTransition:
    balance: float
    newly_resolved: bool
    position_won: bool | None = None
    position_was_open: bool = False
    record_trade_result: bool = False
    recorded_results: tuple[bool, ...] = ()


@dataclass(frozen=True)
class RevalidationDecision:
    accepted: bool
    signal: dict | None
    reason: str | None = None
    eligibility: OrderEligibility | None = None


def market_positions(market: dict) -> list[dict]:
    """Return archived positions followed by the current position."""
    positions = [position for position in market.get("position_history", []) if isinstance(position, dict)]
    current = market.get("position")
    if isinstance(current, dict):
        positions.append(current)
    return positions


def paper_reentry_reason(
    market: dict,
    *,
    now: datetime,
    enabled: bool,
    cooldown_minutes: float,
    max_entries: int,
) -> str | None:
    """Return why another paper entry is blocked, or ``None`` when allowed."""
    current = market.get("position")
    if not isinstance(current, dict):
        return None
    if current.get("status") == "open":
        return "position already open"
    if not enabled:
        return "paper re-entry disabled"
    if len(market_positions(market)) >= max(1, int(max_entries)):
        return "paper re-entry limit"

    raw_closed_at = current.get("closed_at")
    try:
        closed_at = datetime.fromisoformat(str(raw_closed_at).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return "paper re-entry missing close time"
    if closed_at.tzinfo is None:
        closed_at = closed_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    elapsed_minutes = (now.astimezone(timezone.utc) - closed_at.astimezone(timezone.utc)).total_seconds() / 60.0
    if elapsed_minutes < max(0.0, float(cooldown_minutes)):
        return "paper re-entry cooldown"
    return None


def archive_position_for_reentry(market: dict) -> None:
    """Preserve a closed current position before installing a replacement."""
    current = market.get("position")
    if not isinstance(current, dict) or current.get("status") == "open":
        raise ValueError("Only a closed current position can be archived")
    market.setdefault("position_history", []).append(current)
    market["position"] = None


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
    max_relative_spread: float = 1.0,
    min_trade_notional: float = 0.50,
) -> dict | None:
    """Compatibility wrapper returning only an executable signal."""
    decision = revalidate_signal_decision(
        signal,
        market,
        min_ev=min_ev,
        max_price=max_price,
        max_spread=max_spread,
        max_relative_spread=max_relative_spread,
        min_trade_notional=min_trade_notional,
    )
    return decision.signal if decision.accepted else None


def revalidate_signal_decision(
    signal: dict,
    market: dict,
    *,
    min_ev: float,
    max_price: float,
    max_spread: float,
    max_relative_spread: float = 1.0,
    min_trade_notional: float = 0.50,
) -> RevalidationDecision:
    """Revalidate quote, EV, and size against the token's current CLOB book."""
    executable_quote = fetch_executable_quote(str(signal.get("token_id", "")))
    if executable_quote is None:
        return RevalidationDecision(False, None, "book_unavailable")
    quote = Quote(executable_quote.bid, executable_quote.ask)
    spread = quote.ask - quote.bid
    relative_spread = spread / quote.ask if quote.ask > 0 else float("inf")
    if quote.ask >= max_price or spread > max_spread or relative_spread > max_relative_spread:
        return RevalidationDecision(False, None, "price_or_spread")
    fee_rate = market_fee_rate(market)
    ev = fee_adjusted_ev(
        probability=float(signal["p"]),
        price=quote.ask,
        fee_rate=fee_rate,
    )
    if ev < min_ev:
        return RevalidationDecision(False, None, "ev_below_minimum")
    eligibility = evaluate_order_eligibility(
        proposed_notional=float(signal.get("cost", signal.get("amount", 0.0))),
        entry_price=quote.ask,
        fee_rate=fee_rate,
        min_order_size=executable_quote.min_order_size,
        min_trade_notional=min_trade_notional,
    )
    updated = dict(signal)
    updated.update(
        {
            "entry_price": quote.ask,
            "bid_at_entry": quote.bid,
            "spread": round(spread, 4),
            "fee_rate": fee_rate,
            "ev": ev,
            "shares": round(eligibility.shares, 4),
            "min_order_size": executable_quote.min_order_size,
            "tick_size": executable_quote.tick_size,
            "proposed_notional": round(eligibility.proposed_notional, 4),
            "required_notional": (
                round(eligibility.required_notional, 4)
                if eligibility.required_notional is not None
                else None
            ),
            "sizing_decision": "accepted" if eligibility.allowed else eligibility.reason,
        }
    )
    if signal.get("raw_p") is not None:
        updated["raw_ev"] = fee_adjusted_ev(
            probability=float(signal["raw_p"]),
            price=quote.ask,
            fee_rate=fee_rate,
        )
    if not eligibility.allowed:
        return RevalidationDecision(False, updated, eligibility.reason, eligibility)
    return RevalidationDecision(True, updated, eligibility=eligibility)


def record_shadow_signal(
    market: dict,
    signal: dict,
    *,
    recorded_at: str,
    skip_reason: str,
) -> bool:
    """Persist one non-traded diagnostic signal per event/token/side/strategy."""
    key = "|".join(
        (
            str(signal.get("market_id", "")),
            str(signal.get("token_id", "")),
            str(signal.get("outcome_side") or signal.get("side") or "YES").upper(),
            str(signal.get("strategy", "calibrated_mean")),
        )
    )
    shadows = market.setdefault("shadow_signals", [])
    if any(item.get("shadow_key") == key for item in shadows):
        return False
    shadows.append(
        {
            "shadow_key": key,
            "market_id": signal.get("market_id"),
            "token_id": signal.get("token_id"),
            "strategy": signal.get("strategy", "calibrated_mean"),
            "side": str(signal.get("outcome_side") or signal.get("side") or "YES").upper(),
            "probability": signal.get("p", signal.get("probability")),
            "raw_probability": signal.get("raw_p", signal.get("raw_probability")),
            "entry_price": signal.get("entry_price"),
            "bid_at_entry": signal.get("bid_at_entry"),
            "ev": signal.get("ev"),
            "bucket_low": signal.get("bucket_low"),
            "bucket_high": signal.get("bucket_high"),
            "proposed_notional": signal.get("proposed_notional", signal.get("cost")),
            "required_notional": signal.get("required_notional"),
            "min_order_size": signal.get("min_order_size"),
            "skip_reason": skip_reason,
            "forecast_source": signal.get("forecast_src", signal.get("forecast_source")),
            "forecast_calibration_n": signal.get("forecast_calibration_n", 0),
            "recorded_at": recorded_at,
            "eventual_outcome": None,
            "settled_at": None,
        }
    )
    return True


def settle_shadow_signals(
    market: dict,
    *,
    winning_market_id: str,
    resolved_at: str,
) -> int:
    """Resolve diagnostic signals without changing paper cash or trade results."""
    settled = 0
    for signal in market.get("shadow_signals", []):
        if signal.get("eventual_outcome") is not None:
            continue
        yes_won = str(signal.get("market_id")) == str(winning_market_id)
        side = str(signal.get("side", "YES")).upper()
        won = yes_won if side == "YES" else not yes_won
        signal["eventual_outcome"] = "win" if won else "loss"
        signal["settled_at"] = resolved_at
        settled += 1
    return settled


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


def settle_paper_market(
    market: dict,
    *,
    yes_won: bool | None = None,
    winning_market_id: str | None = None,
    balance: float,
    resolved_at: str,
    actual_temp: float | None = None,
) -> ResolutionTransition:
    """Record eventual market settlement without double-counting an early exit."""
    if market.get("status") == "resolved" or market.get("resolved") is True:
        return ResolutionTransition(balance=balance, newly_resolved=False)

    positions = market_positions(market)
    if not positions:
        return ResolutionTransition(balance=balance, newly_resolved=False)

    market["status"] = "resolved"
    market["resolved"] = True
    market["resolved_at"] = resolved_at
    market["winning_market_id"] = winning_market_id
    if yes_won is not None:
        market["market_yes_won"] = bool(yes_won)
    if actual_temp is not None:
        market["actual_temp"] = float(actual_temp)
        market["actual_temp_observed_at"] = resolved_at

    recorded_results: list[bool] = []
    any_open = False
    last_position_won: bool | None = None
    for position in positions:
        selected_yes_won = (
            str(position.get("market_id")) == str(winning_market_id)
            if winning_market_id is not None
            else bool(yes_won)
        )
        side = str(position.get("outcome_side") or position.get("side") or "YES").upper()
        position_won = selected_yes_won if side == "YES" else not selected_yes_won
        last_position_won = position_won
        position_was_open = position.get("status") == "open"
        any_open = any_open or position_was_open
        position["eventual_outcome"] = "win" if position_won else "loss"
        position["would_have_won_at_resolution"] = position_won
        position["settled_at"] = resolved_at

        if position_was_open:
            cost = float(position.get("cost", position.get("amount", 0.0)))
            shares = float(position.get("shares", 0.0))
            payout = shares if position_won else 0.0
            pnl = round(payout - cost, 2)
            balance = round(balance + payout, 2)
            position.update(
                {
                    "exit_price": 1.0 if position_won else 0.0,
                    "pnl": pnl,
                    "close_reason": "resolved",
                    "closed_at": resolved_at,
                    "status": "closed",
                }
            )

        if not bool(position.get("trade_result_recorded")) and position.get("pnl") is not None:
            position["trade_result_recorded"] = True
            recorded_results.append(float(position["pnl"]) >= 0.0)

    market["resolved_outcome"] = "win" if last_position_won else "loss"
    market["pnl"] = sum(float(position.get("pnl") or 0.0) for position in positions)

    return ResolutionTransition(
        balance=balance,
        newly_resolved=True,
        position_won=last_position_won,
        position_was_open=any_open,
        record_trade_result=bool(recorded_results),
        recorded_results=tuple(recorded_results),
    )
