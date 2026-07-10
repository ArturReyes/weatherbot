"""Chronological, no-lookahead evaluation for stored paper-trade decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from math import ceil


@dataclass(frozen=True)
class PromotionPolicy:
    holdout_fraction: float = 0.25
    min_holdout_trades: int = 30
    min_brier_samples: int = 30
    min_realized_roi: float = 0.0
    max_brier_score: float = 0.25
    max_drawdown_pct: float = 0.10

    @classmethod
    def from_mapping(cls, config: dict) -> "PromotionPolicy":
        return cls(
            holdout_fraction=float(config.get("validation_holdout_fraction", 0.25)),
            min_holdout_trades=int(config.get("promotion_min_holdout_trades", 30)),
            min_brier_samples=int(config.get("promotion_min_brier_samples", 30)),
            min_realized_roi=float(config.get("promotion_min_realized_roi", 0.0)),
            max_brier_score=float(config.get("promotion_max_brier_score", 0.25)),
            max_drawdown_pct=float(config.get("promotion_max_drawdown_pct", 0.10)),
        )


@dataclass(frozen=True)
class TradeObservation:
    strategy: str
    closed_at: datetime
    pnl: float
    cost: float
    probability: float | None
    resolved_outcome: int | None
    expected_ev: float | None
    exit_price: float | None
    entry_price: float | None
    close_reason: str | None


@dataclass(frozen=True)
class StrategyMetrics:
    strategy: str
    trades: int
    wins: int
    total_cost: float
    pnl: float
    realized_roi: float | None
    brier_score: float | None
    brier_samples: int
    mean_expected_ev: float | None
    max_drawdown_pct: float
    post_entry_price_change: float | None
    post_entry_price_samples: int


@dataclass(frozen=True)
class PromotionDecision:
    ready: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class OutOfSampleReport:
    training: dict[str, StrategyMetrics]
    holdout: dict[str, StrategyMetrics]
    decision: PromotionDecision
    holdout_fraction: float


def chronological_out_of_sample_report(
    markets: list[dict],
    *,
    policy: PromotionPolicy,
    bankroll: float,
    required_strategies: tuple[str, ...] = (),
) -> OutOfSampleReport:
    """Evaluate executed signals using only their later realised outcomes.

    The final chronological fraction is holdout data.  This is not a synthetic
    full-market backtest: it reports the bot's actual, timestamped paper
    decisions and never lets a later resolution influence an earlier entry.
    """
    observations = sorted(_trade_observations(markets), key=lambda item: item.closed_at)
    if not observations:
        reasons = tuple(f"{strategy}: no closed paper positions" for strategy in required_strategies)
        decision = PromotionDecision(False, reasons or ("no closed paper positions",))
        return OutOfSampleReport({}, {}, decision, policy.holdout_fraction)

    holdout_count = max(1, ceil(len(observations) * policy.holdout_fraction))
    if holdout_count >= len(observations):
        training_observations = []
        holdout_observations = observations
    else:
        training_observations = observations[:-holdout_count]
        holdout_observations = observations[-holdout_count:]

    training = _metrics_by_strategy(training_observations, bankroll)
    holdout = _metrics_by_strategy(holdout_observations, bankroll)
    return OutOfSampleReport(
        training=training,
        holdout=holdout,
        decision=_promotion_decision(holdout, policy, required_strategies),
        holdout_fraction=policy.holdout_fraction,
    )


def _trade_observations(markets: list[dict]) -> list[TradeObservation]:
    observations: list[TradeObservation] = []
    for market in markets:
        position = market.get("position") or {}
        if position.get("status") != "closed" or position.get("pnl") is None:
            continue
        closed_at = _timestamp(position.get("closed_at") or market.get("closed_at"))
        if closed_at is None:
            continue
        probability = _number(position.get("p", position.get("probability")))
        resolved_outcome = _resolved_outcome(market)
        observations.append(
            TradeObservation(
                strategy=str(position.get("strategy", "calibrated_mean")),
                closed_at=closed_at,
                pnl=float(position["pnl"]),
                cost=max(0.0, float(position.get("cost", position.get("amount", 0.0)))),
                probability=probability,
                resolved_outcome=resolved_outcome,
                expected_ev=_number(position.get("ev")),
                exit_price=_number(position.get("exit_price")),
                entry_price=_number(position.get("entry_price")),
                close_reason=position.get("close_reason"),
            )
        )
    return observations


def _metrics_by_strategy(
    observations: list[TradeObservation],
    bankroll: float,
) -> dict[str, StrategyMetrics]:
    strategies = sorted({observation.strategy for observation in observations})
    return {
        strategy: _metrics(strategy, [item for item in observations if item.strategy == strategy], bankroll)
        for strategy in strategies
    }


def _metrics(strategy: str, observations: list[TradeObservation], bankroll: float) -> StrategyMetrics:
    total_cost = sum(item.cost for item in observations)
    pnl = sum(item.pnl for item in observations)
    resolved = [item for item in observations if item.probability is not None and item.resolved_outcome is not None]
    brier_score = (
        sum((item.probability - item.resolved_outcome) ** 2 for item in resolved) / len(resolved)
        if resolved
        else None
    )
    expected_values = [item.expected_ev for item in observations if item.expected_ev is not None]
    post_entry = [
        item.exit_price - item.entry_price
        for item in observations
        if item.close_reason != "resolved"
        and item.exit_price is not None
        and item.entry_price is not None
    ]
    return StrategyMetrics(
        strategy=strategy,
        trades=len(observations),
        wins=sum(1 for item in observations if item.pnl > 0),
        total_cost=round(total_cost, 4),
        pnl=round(pnl, 4),
        realized_roi=round(pnl / total_cost, 6) if total_cost > 0 else None,
        brier_score=round(brier_score, 6) if brier_score is not None else None,
        brier_samples=len(resolved),
        mean_expected_ev=round(sum(expected_values) / len(expected_values), 6) if expected_values else None,
        max_drawdown_pct=round(_max_drawdown_pct(observations, bankroll), 6),
        post_entry_price_change=round(sum(post_entry) / len(post_entry), 6) if post_entry else None,
        post_entry_price_samples=len(post_entry),
    )


def _promotion_decision(
    holdout: dict[str, StrategyMetrics],
    policy: PromotionPolicy,
    required_strategies: tuple[str, ...],
) -> PromotionDecision:
    if not holdout:
        return PromotionDecision(False, ("no holdout trades",))
    reasons: list[str] = []
    strategies = required_strategies or tuple(holdout)
    for strategy in strategies:
        metrics = holdout.get(strategy)
        if metrics is None:
            reasons.append(f"{strategy}: no holdout trades")
            continue
        prefix = f"{strategy}:"
        if metrics.trades < policy.min_holdout_trades:
            reasons.append(f"{prefix} {metrics.trades}/{policy.min_holdout_trades} holdout trades")
        if metrics.brier_samples < policy.min_brier_samples:
            reasons.append(f"{prefix} {metrics.brier_samples}/{policy.min_brier_samples} resolved probability samples")
        elif metrics.brier_score is not None and metrics.brier_score > policy.max_brier_score:
            reasons.append(f"{prefix} Brier {metrics.brier_score:.3f} exceeds {policy.max_brier_score:.3f}")
        if metrics.realized_roi is None or metrics.realized_roi < policy.min_realized_roi:
            reasons.append(f"{prefix} realised ROI below {policy.min_realized_roi:.1%}")
        if metrics.max_drawdown_pct > policy.max_drawdown_pct:
            reasons.append(f"{prefix} drawdown {metrics.max_drawdown_pct:.1%} exceeds {policy.max_drawdown_pct:.1%}")
    return PromotionDecision(not reasons, tuple(reasons))


def _max_drawdown_pct(observations: list[TradeObservation], bankroll: float) -> float:
    equity = max(float(bankroll), 1e-9)
    peak = equity
    maximum = 0.0
    for observation in observations:
        equity += observation.pnl
        peak = max(peak, equity)
        maximum = max(maximum, (peak - equity) / peak if peak > 0 else 1.0)
    return maximum


def _resolved_outcome(market: dict) -> int | None:
    outcome = market.get("resolved_outcome")
    if outcome == "win":
        return 1
    if outcome == "loss":
        return 0
    return None


def _number(value: object) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _timestamp(value: object) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
