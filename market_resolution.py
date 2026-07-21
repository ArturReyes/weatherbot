"""Pure Polymarket settlement and calibration-observation primitives."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass


TAIL_LOW = -999.0
TAIL_HIGH = 999.0


@dataclass(frozen=True)
class SettledBucket:
    market_id: str
    low: float
    high: float
    representative_temp: float | None

    @property
    def bounded(self) -> bool:
        return self.representative_temp is not None


@dataclass(frozen=True)
class WinningBucketDecision:
    bucket: SettledBucket | None
    reason: str | None = None


def winning_bucket_from_outcomes(
    outcomes: list[dict],
    *,
    allow_legacy_price: bool = False,
) -> WinningBucketDecision:
    """Return the unique bucket whose settled YES token is worth one dollar."""
    winners: list[dict] = []
    for outcome in outcomes:
        price = _settlement_yes_price(outcome, allow_legacy_price=allow_legacy_price)
        if price is not None and price >= 0.95:
            winners.append(outcome)
    if not winners:
        return WinningBucketDecision(None, "unresolved")
    if len(winners) != 1:
        return WinningBucketDecision(None, "ambiguous_winner")

    winner = winners[0]
    raw_range = winner.get("range")
    if not isinstance(raw_range, (list, tuple)) or len(raw_range) != 2:
        return WinningBucketDecision(None, "missing_bucket_range")
    try:
        low, high = float(raw_range[0]), float(raw_range[1])
    except (TypeError, ValueError):
        return WinningBucketDecision(None, "invalid_bucket_range")
    if not math.isfinite(low) or not math.isfinite(high) or low > high:
        return WinningBucketDecision(None, "invalid_bucket_range")

    representative = None
    if low > TAIL_LOW and high < TAIL_HIGH:
        representative = (low + high) / 2.0
    return WinningBucketDecision(
        SettledBucket(
            market_id=str(winner.get("market_id") or winner.get("id") or ""),
            low=low,
            high=high,
            representative_temp=representative,
        )
    )


def calibration_fields(
    bucket: SettledBucket,
    *,
    provider: str,
    station: str,
    validated_at: str,
) -> dict:
    """Build auditable fields for a bounded settlement observation."""
    if not bucket.bounded:
        raise ValueError("Open-ended winning buckets cannot calibrate temperature")
    return {
        "calibration_temp": bucket.representative_temp,
        "calibration_bucket_low": bucket.low,
        "calibration_bucket_high": bucket.high,
        "calibration_source": "polymarket_winning_bucket",
        "resolution_provider": provider,
        "resolution_station": station.upper(),
        "calibration_validated_at": validated_at,
    }


def _settlement_yes_price(outcome: dict, *, allow_legacy_price: bool) -> float | None:
    if outcome.get("settlement_yes_price") is not None:
        try:
            return float(outcome["settlement_yes_price"])
        except (TypeError, ValueError):
            return None
    raw_prices = outcome.get("outcomePrices")
    if raw_prices is not None:
        if isinstance(raw_prices, str):
            try:
                raw_prices = json.loads(raw_prices)
            except (TypeError, json.JSONDecodeError):
                return None
        if isinstance(raw_prices, (list, tuple)) and raw_prices:
            try:
                return float(raw_prices[0])
            except (TypeError, ValueError):
                return None
    if allow_legacy_price and outcome.get("price") is not None:
        try:
            return float(outcome["price"])
        except (TypeError, ValueError):
            return None
    return None
