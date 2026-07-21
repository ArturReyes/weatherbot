"""Time-safe calibration primitives for forecast error models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


LEAD_TIME_BUCKETS = (
    (0.0, 6.0, "0_6h"),
    (6.0, 12.0, "6_12h"),
    (12.0, 24.0, "12_24h"),
    (24.0, 48.0, "24_48h"),
    (48.0, 72.0, "48_72h"),
)


@dataclass(frozen=True)
class BiasEstimate:
    """Forecast mean-error estimate.

    ``raw_bias`` is forecast - actual. Positive means the source has been too
    warm/high and should be subtracted from future forecasts.
    """

    bias: float
    raw_bias: float
    n: int


def _timestamp(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def lead_time_bucket(snapshot_ts: object, event_end_ts: object) -> str | None:
    """Bucket the forecast lead time between snapshot and market resolution."""
    snapshot_time = _timestamp(snapshot_ts)
    event_time = _timestamp(event_end_ts)
    if snapshot_time is None or event_time is None:
        return None

    lead_hours = (event_time - snapshot_time).total_seconds() / 3600
    if lead_hours < 0:
        return None

    for low, high, label in LEAD_TIME_BUCKETS:
        if low <= lead_hours < high:
            return label
    return "72h_plus"


def select_calibration_snapshot(
    market: dict,
    source: str,
    *,
    lead_bucket: str | None = None,
) -> dict | None:
    """Select a time-safe forecast, optionally from one lead-time bucket."""
    candidates = [
        snapshot
        for snapshot in market.get("forecast_snapshots", [])
        if snapshot.get(source) is not None and _timestamp(snapshot.get("ts")) is not None
    ]
    if not candidates:
        return None

    position = market.get("position") or {}
    cutoff = _timestamp(position.get("opened_at") or position.get("entered_at"))
    if cutoff is not None:
        candidates = [item for item in candidates if _timestamp(item["ts"]) <= cutoff]
    if lead_bucket is not None:
        candidates = [
            item
            for item in candidates
            if lead_time_bucket(item.get("ts"), _event_end_ts(market)) == lead_bucket
        ]
    if not candidates:
        return None

    if cutoff is None and lead_bucket is None:
        return min(candidates, key=lambda item: _timestamp(item["ts"]))
    return max(candidates, key=lambda item: _timestamp(item["ts"]))


def _has_calibration_observation(market: dict) -> bool:
    return (
        market.get("calibration_temp") is not None
        and market.get("calibration_source") == "polymarket_winning_bucket"
        and (
        market.get("resolved") is True
        or market.get("status") in {"closed", "resolved"}
        )
    )


def calibration_errors(
    markets: list[dict],
    *,
    city: str,
    source: str,
    lead_bucket: str | None = None,
) -> list[float]:
    errors: list[float] = []
    for market in markets:
        if (
            market.get("city") != city
            or not _has_calibration_observation(market)
        ):
            continue
        snapshot = select_calibration_snapshot(
            market,
            source,
            lead_bucket=lead_bucket,
        )
        if snapshot is not None:
            errors.append(float(snapshot[source]) - float(market["calibration_temp"]))
    return errors


def _event_end_ts(market: dict) -> object:
    return (
        market.get("event_end_date")
        or market.get("eventEndDate")
        or market.get("endDate")
    )


def _resolved_at(market: dict) -> datetime | None:
    return _timestamp(
        market.get("resolved_at")
        or market.get("closed_at")
        or market.get("calibration_validated_at")
        or market.get("actual_temp_observed_at")
    )


def decaying_mean_error(
    markets: list[dict],
    *,
    city: str,
    source: str,
    lead_bucket: str | None = None,
    as_of: object = None,
    decay: float = 0.97,
    prior_strength: float = 20.0,
) -> BiasEstimate:
    """Estimate forecast - actual error with recent samples weighted higher.

    The function only uses the calibration snapshot selected by
    ``select_calibration_snapshot``, so post-entry forecast updates cannot leak
    into the residual estimate.
    """
    if not 0 < decay <= 1:
        raise ValueError("decay must be in (0, 1]")
    if prior_strength < 0:
        raise ValueError("prior_strength must be >= 0")

    cutoff = _timestamp(as_of)
    samples: list[tuple[datetime, float]] = []

    for market in markets:
        if (
            market.get("city") != city
            or not _has_calibration_observation(market)
        ):
            continue

        resolved_at = _resolved_at(market)
        if cutoff is not None and resolved_at is not None and resolved_at > cutoff:
            continue

        snapshot = select_calibration_snapshot(
            market,
            source,
            lead_bucket=lead_bucket,
        )
        if snapshot is None:
            continue

        snapshot_time = _timestamp(snapshot.get("ts"))
        if snapshot_time is None:
            continue
        if cutoff is not None and snapshot_time > cutoff:
            continue

        error = float(snapshot[source]) - float(market["calibration_temp"])
        samples.append((snapshot_time, error))

    if not samples:
        return BiasEstimate(bias=0.0, raw_bias=0.0, n=0)

    samples.sort(key=lambda item: item[0])
    weighted_sum = 0.0
    weight_sum = 0.0
    sample_count = len(samples)
    for index, (_, error) in enumerate(samples):
        weight = decay ** (sample_count - index - 1)
        weighted_sum += error * weight
        weight_sum += weight

    raw_bias = weighted_sum / weight_sum
    shrinkage = sample_count / (sample_count + prior_strength)
    return BiasEstimate(
        bias=raw_bias * shrinkage,
        raw_bias=raw_bias,
        n=sample_count,
    )


def bias_adjusted_forecast(
    forecast: float,
    bias: float,
    *,
    max_correction: float | None = None,
) -> float:
    """Apply a forecast - actual bias correction to a raw forecast."""
    correction = float(bias)
    if max_correction is not None:
        if max_correction < 0:
            raise ValueError("max_correction must be >= 0")
        correction = max(-max_correction, min(max_correction, correction))
    return float(forecast) - correction


def rmse_sigma(errors: list[float], *, floor: float = 0.0) -> float:
    if not errors:
        raise ValueError("At least one calibration error is required")
    rmse = math.sqrt(sum(error * error for error in errors) / len(errors))
    return max(float(floor), rmse)


def regularized_sigma(
    errors: list[float],
    *,
    prior_sigma: float,
    prior_strength: float,
    floor: float = 0.0,
) -> float:
    """Shrink small-sample RMSE variance toward a conservative prior."""
    if not errors:
        raise ValueError("At least one calibration error is required")
    if prior_sigma <= 0 or prior_strength < 0:
        raise ValueError("prior_sigma must be positive and prior_strength non-negative")
    sample_variance = sum(error * error for error in errors) / len(errors)
    weight = len(errors) / (len(errors) + prior_strength)
    variance = weight * sample_variance + (1.0 - weight) * prior_sigma * prior_sigma
    return max(float(floor), math.sqrt(variance))
