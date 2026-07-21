"""Pure order-size eligibility rules shared by paper and live execution."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OrderEligibility:
    """Result of applying operator and exchange minimum-order constraints."""

    allowed: bool
    reason: str | None
    proposed_notional: float
    required_notional: float | None
    shares: float
    min_order_size: float | None
    entry_cost_per_share: float | None


def entry_cost_per_share(*, price: float, fee_rate: float) -> float | None:
    """Return the fee-adjusted cash required to buy one outcome share."""
    if not 0 < price < 1 or not 0 <= fee_rate < 1:
        return None
    return price + fee_rate * price * (1.0 - price)


def evaluate_order_eligibility(
    *,
    proposed_notional: float,
    entry_price: float,
    fee_rate: float,
    min_order_size: float | None,
    min_trade_notional: float,
) -> OrderEligibility:
    """Decide whether a risk-sized order is executable without increasing it.

    ``proposed_notional`` remains authoritative.  The function never rounds an
    undersized proposal up to either the operator floor or exchange minimum.
    """
    proposed = max(0.0, float(proposed_notional))
    floor = max(0.0, float(min_trade_notional))
    cost_per_share = entry_cost_per_share(price=float(entry_price), fee_rate=float(fee_rate))
    if cost_per_share is None:
        return OrderEligibility(
            False,
            "invalid_entry_cost",
            proposed,
            None,
            0.0,
            min_order_size,
            None,
        )

    try:
        exchange_minimum = float(min_order_size) if min_order_size is not None else 0.0
    except (TypeError, ValueError):
        exchange_minimum = 0.0
    if exchange_minimum <= 0:
        return OrderEligibility(
            False,
            "missing_min_order_size",
            proposed,
            None,
            proposed / cost_per_share,
            None,
            cost_per_share,
        )

    required = max(floor, exchange_minimum * cost_per_share)
    shares = proposed / cost_per_share
    if proposed + 1e-9 < required:
        return OrderEligibility(
            False,
            "below_order_minimum",
            proposed,
            required,
            shares,
            exchange_minimum,
            cost_per_share,
        )
    return OrderEligibility(
        True,
        None,
        proposed,
        required,
        shares,
        exchange_minimum,
        cost_per_share,
    )
