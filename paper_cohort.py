"""Pure lifecycle helpers for archiving and resetting paper cohorts."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from paper_trading import market_positions


def parse_timestamp(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def cohort_position_records(
    markets: Iterable[dict],
    *,
    started_at: str | None,
) -> list[tuple[dict, dict]]:
    """Return market/position pairs opened in the selected evaluation cohort."""
    cutoff = parse_timestamp(started_at)
    records: list[tuple[dict, dict]] = []
    for market in markets:
        for position in market_positions(market):
            opened_at = parse_timestamp(position.get("opened_at") or position.get("entered_at"))
            if cutoff is not None and (opened_at is None or opened_at < cutoff):
                continue
            records.append((market, position))
    return records


def ensure_paper_reset_allowed(markets: Iterable[dict], state: dict) -> None:
    open_positions = [
        position
        for market in markets
        for position in market_positions(market)
        if position.get("status") == "open"
    ]
    if open_positions:
        raise ValueError(f"cannot reset paper cohort with {len(open_positions)} open position(s)")
    if (
        state.get("paper_cohort_id")
        and int(state.get("total_trades", 0)) == 0
        and int(state.get("wins", 0)) == 0
        and int(state.get("losses", 0)) == 0
    ):
        raise ValueError("current paper cohort is already fresh")


def build_cohort_archive(
    *,
    markets: Iterable[dict],
    state: dict,
    cohort_id: str,
    ended_at: str,
    evaluation_started_at: str | None,
) -> dict:
    trades = []
    for market in markets:
        for position in market_positions(market):
            trades.append(
                {
                    "city": market.get("city"),
                    "city_name": market.get("city_name"),
                    "date": market.get("date"),
                    "market_status": market.get("status"),
                    "actual_temp": market.get("actual_temp"),
                    "position": position,
                }
            )
    return {
        "schema_version": 1,
        "cohort_id": cohort_id,
        "evaluation_started_at": evaluation_started_at,
        "ended_at": ended_at,
        "state": state,
        "summary": {
            "positions": len(trades),
            "open_positions": sum(
                trade["position"].get("status") == "open" for trade in trades
            ),
            "resolved_positions": sum(
                trade["position"].get("eventual_outcome") is not None for trade in trades
            ),
            "realized_pnl": round(
                sum(float(trade["position"].get("pnl") or 0.0) for trade in trades),
                2,
            ),
        },
        "trades": trades,
        "preserved_data": ["data/markets", "data/calibration.json"],
    }


def build_fresh_paper_state(
    *,
    previous_state: dict,
    bankroll: float,
    cohort_id: str,
    started_at: str,
    archive_path: str,
) -> dict:
    archives = list(previous_state.get("paper_cohort_archives", []))
    archives.append(archive_path)
    fresh = dict(previous_state)
    fresh.update(
        {
            "balance": float(bankroll),
            "starting_balance": float(bankroll),
            "total_trades": 0,
            "wins": 0,
            "losses": 0,
            "peak_balance": float(bankroll),
            "paper_cohort_id": cohort_id,
            "evaluation_started_at": started_at,
            "paper_cohort_archives": archives,
        }
    )
    return fresh


def mark_market_positions_legacy(market: dict, *, cohort_id: str) -> bool:
    """Prevent later settlement of an archived trade from changing fresh counters."""
    changed = False
    for position in market_positions(market):
        if position.get("paper_cohort_id") != cohort_id:
            position["paper_cohort_id"] = cohort_id
            changed = True
        if position.get("pnl") is not None and not position.get("trade_result_recorded"):
            position["trade_result_recorded"] = True
            changed = True
    return changed
