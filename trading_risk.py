"""Pure validation and risk math for live weather trading."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone


WEATHER_TAKER_FEE_RATE = 0.05


@dataclass(frozen=True)
class ContractValidation:
    valid: bool
    reason: str | None = None


@dataclass(frozen=True)
class RiskLimits:
    max_total_exposure_pct: float
    max_event_exposure_pct: float
    max_daily_loss_pct: float
    max_open_positions: int
    max_signal_age_seconds: float


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str | None = None
    total_exposure: float = 0.0
    event_exposure: float = 0.0
    daily_loss: float = 0.0
    active_positions: int = 0


ACTIVE_POSITION_STATUSES = frozenset({
    "submitting", "pending", "unknown", "open", "exit_pending", "exit_unknown",
    "redeeming", "redemption_unknown", "redemption_confirmed", "missing", "unmanaged",
})


def assess_trade_risk(
    state: dict,
    *,
    size_usdc: float,
    city_slug: str,
    date_str: str,
    signal_created_at: float,
    bankroll: float,
    limits: RiskLimits,
    now_ts: float,
) -> RiskDecision:
    """Evaluate projected portfolio state before persisting an order intent."""
    positions = state.get("positions", [])
    active = [position for position in positions if position.get("status") in ACTIVE_POSITION_STATUSES]
    total_exposure = sum(_position_exposure(position) for position in active)
    event_exposure = sum(
        _position_exposure(position)
        for position in active
        if position.get("city_slug") == city_slug
        and (position.get("date") or position.get("date_str")) == date_str
    )
    daily_loss = _realized_loss_today(positions, now_ts)
    metrics = {
        "total_exposure": total_exposure,
        "event_exposure": event_exposure,
        "daily_loss": daily_loss,
        "active_positions": len(active),
    }

    age = now_ts - signal_created_at
    if age < -5 or age > limits.max_signal_age_seconds:
        return RiskDecision(False, "stale_signal", **metrics)
    if bankroll <= 0 or size_usdc <= 0:
        return RiskDecision(False, "invalid_trade_size", **metrics)
    if daily_loss >= bankroll * limits.max_daily_loss_pct:
        return RiskDecision(False, "daily_loss_limit", **metrics)
    if len(active) >= limits.max_open_positions:
        return RiskDecision(False, "open_position_limit", **metrics)
    if total_exposure + size_usdc > bankroll * limits.max_total_exposure_pct:
        return RiskDecision(False, "total_exposure_limit", **metrics)
    if event_exposure + size_usdc > bankroll * limits.max_event_exposure_pct:
        return RiskDecision(False, "event_exposure_limit", **metrics)
    return RiskDecision(True, **metrics)


def _position_exposure(position: dict) -> float:
    amount = max(0.0, float(position.get("amount") or position.get("cost") or 0.0))
    if position.get("status") in {"submitting", "pending", "unknown"}:
        requested = max(0.0, float(position.get("requested_amount") or 0.0))
        return max(amount, requested)
    return amount


def _realized_loss_today(positions: list[dict], now_ts: float) -> float:
    today = datetime.fromtimestamp(now_ts, tz=timezone.utc).date()
    loss = 0.0
    for position in positions:
        if position.get("status") != "closed" or position.get("pnl") is None:
            continue
        raw_timestamp = position.get("exited_at") or position.get("closed_at")
        try:
            exited = datetime.fromisoformat(str(raw_timestamp).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            continue
        if exited.tzinfo is None:
            exited = exited.replace(tzinfo=timezone.utc)
        if exited.astimezone(timezone.utc).date() == today:
            loss += max(0.0, -float(position["pnl"]))
    return loss


def extract_market_date(slug: str) -> str | None:
    iso = re.search(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)", slug)
    if iso:
        try:
            return datetime.strptime(iso.group(0), "%Y-%m-%d").strftime("%Y-%m-%d")
        except ValueError:
            return None

    named = re.search(
        r"-on-(january|february|march|april|may|june|july|august|"
        r"september|october|november|december)-(\d{1,2})-(\d{4})(?:-|$)",
        slug,
        re.IGNORECASE,
    )
    if not named:
        return None
    try:
        parsed = datetime.strptime("-".join(named.groups()), "%B-%d-%Y")
    except ValueError:
        return None
    return parsed.strftime("%Y-%m-%d")


def contract_matches_strategy(
    market: dict,
    *,
    city_name: str,
    station: str,
    unit: str,
    date_str: str,
) -> ContractValidation:
    question = str(market.get("question", ""))
    description = str(market.get("description", ""))
    slug = str(market.get("slug", ""))
    question_lower = question.lower()
    description_lower = description.lower()

    if "highest temperature" not in question_lower or "highest temperature" not in description_lower:
        return ContractValidation(False, "not_high_temperature")
    if city_name.lower() not in question_lower:
        return ContractValidation(False, "city_mismatch")
    expected_symbol = f"°{unit.upper()}"
    if expected_symbol not in question.upper():
        return ContractValidation(False, "unit_mismatch")
    resolution_unit = "degrees fahrenheit" if unit.upper() == "F" else "degrees celsius"
    if resolution_unit not in description_lower:
        return ContractValidation(False, "resolution_unit_mismatch")
    if f"site={station.lower()}" not in description_lower:
        return ContractValidation(False, "station_mismatch")
    if "noaa" not in description_lower or "highest reading" not in description_lower:
        return ContractValidation(False, "resolution_source_mismatch")
    if extract_market_date(slug) != date_str:
        return ContractValidation(False, "date_mismatch")
    if not _description_matches_date(description_lower, date_str):
        return ContractValidation(False, "resolution_date_mismatch")
    if market.get("enableOrderBook") is not True:
        return ContractValidation(False, "orderbook_disabled")
    if market.get("acceptingOrders") is not True:
        return ContractValidation(False, "orders_not_accepted")
    return ContractValidation(True)


def market_fee_rate(market: dict) -> float:
    if market.get("feesEnabled") is False:
        return 0.0
    schedule = market.get("feeSchedule")
    if isinstance(schedule, dict):
        try:
            rate = float(schedule["rate"])
        except (KeyError, TypeError, ValueError):
            rate = WEATHER_TAKER_FEE_RATE
        if 0 <= rate < 1:
            return rate
    return WEATHER_TAKER_FEE_RATE if market.get("feesEnabled") else 0.0


def fee_adjusted_ev(*, probability: float, price: float, fee_rate: float) -> float:
    total_cost = _total_entry_cost(price, fee_rate)
    if total_cost <= 0 or total_cost >= 1:
        return 0.0
    return round((probability - total_cost) / total_cost, 4)


def fee_adjusted_kelly(*, probability: float, price: float, fee_rate: float) -> float:
    total_cost = _total_entry_cost(price, fee_rate)
    if total_cost <= 0 or total_cost >= 1:
        return 0.0
    return round(max(0.0, (probability - total_cost) / (1.0 - total_cost)), 4)


def _total_entry_cost(price: float, fee_rate: float) -> float:
    if price <= 0 or price >= 1 or fee_rate < 0:
        return 0.0
    return price + fee_rate * price * (1.0 - price)


def _description_matches_date(description: str, date_str: str) -> bool:
    try:
        value = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return False
    day = value.day
    year = value.year
    short_year = str(year)[2:]
    month = value.strftime("%B").lower()
    short_month = value.strftime("%b").lower()
    candidates = (
        f"{day} {short_month} '{short_year}",
        f"{day} {month} {year}",
        f"{month} {day}, {year}",
        f"{month} {day} {year}",
    )
    return any(candidate in description for candidate in candidates)
