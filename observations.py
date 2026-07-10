"""Pure daily station-observation completeness checks.

Near-resolution trades are only safe when the bot has observed the station for
the whole local market day.  A current METAR reading by itself is not proof
that an earlier, higher reading did not occur.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class DailyObservedHigh:
    high: float | None
    complete: bool
    first_observation_at: datetime | None = None
    last_observation_at: datetime | None = None


def daily_observed_high(
    observations: list[dict],
    *,
    market_date: str,
    timezone_name: str,
    now: datetime,
    start_grace_minutes: float = 90.0,
    max_gap_minutes: float = 90.0,
) -> DailyObservedHigh:
    """Return the local-day high and whether coverage is continuous enough.

    The caller supplies persisted ``{"ts", "value"}`` observations.  The
    function intentionally fails closed for a restart after the morning
    coverage window or any long collection gap.
    """
    try:
        target_date = date.fromisoformat(market_date)
        tz = ZoneInfo(timezone_name)
    except (TypeError, ValueError):
        return DailyObservedHigh(None, False)

    now_local = _as_local(now, tz)
    if now_local.date() != target_date:
        return DailyObservedHigh(None, False)

    samples: list[tuple[datetime, float]] = []
    for observation in observations:
        observed_at = _parse_timestamp(observation.get("ts"), tz)
        value = observation.get("value", observation.get("metar"))
        if observed_at is None or value is None or observed_at.date() != target_date:
            continue
        try:
            samples.append((observed_at, float(value)))
        except (TypeError, ValueError):
            continue

    if not samples:
        return DailyObservedHigh(None, False)

    samples.sort(key=lambda item: item[0])
    first_at = samples[0][0]
    last_at = samples[-1][0]
    local_midnight = datetime.combine(target_date, datetime.min.time(), tzinfo=tz)
    start_deadline = local_midnight + timedelta(minutes=start_grace_minutes)
    max_gap = timedelta(minutes=max_gap_minutes)
    complete = first_at <= start_deadline

    previous_at = first_at
    for observed_at, _ in samples[1:]:
        if observed_at - previous_at > max_gap:
            complete = False
            break
        previous_at = observed_at
    if now_local - last_at > max_gap:
        complete = False

    return DailyObservedHigh(
        high=max(value for _, value in samples),
        complete=complete,
        first_observation_at=first_at,
        last_observation_at=last_at,
    )


def _as_local(value: datetime, timezone_value: ZoneInfo) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone_value)
    return value.astimezone(timezone_value)


def _parse_timestamp(value: object, timezone_value: ZoneInfo) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    return _as_local(parsed, timezone_value)
