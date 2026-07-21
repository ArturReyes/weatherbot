"""Pure strategy candidate generation for weather temperature markets.

This module intentionally has no network, file, or order-execution code. Paper
and live trading pass market/weather context in and receive ranked candidates
out. That keeps structural edges testable and shared across execution modes.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from trading_risk import fee_adjusted_ev


YES = "YES"
NO = "NO"

CALIBRATED_MEAN = "calibrated_mean"
NEAR_LOCK = "near_lock"
UNDERDISPERSION_TAIL = "underdispersion_tail"
MODEL_LAG = "model_lag"

EXIT_STANDARD = "standard"
EXIT_HOLD_TO_RESOLUTION = "hold_to_resolution"


@dataclass(frozen=True)
class StrategyConfig:
    min_ev: float = 0.10
    max_price: float = 0.45
    max_slippage: float = 0.03
    max_relative_spread: float = 0.25
    strategy_calibrated_mean_enabled: bool = True
    strategy_near_lock_enabled: bool = True
    strategy_underdispersion_enabled: bool = False
    strategy_model_lag_enabled: bool = True
    enable_no_trades: bool = False
    calibrated_mean_min_samples: int = 30
    standard_price_stop_enabled: bool = False
    standard_stop_loss_fraction: float = 0.20
    standard_trailing_activation_return: float = 0.20
    standard_take_profit_enabled: bool = True
    standard_take_profit_24_48_price: float = 0.85
    standard_take_profit_48_plus_price: float = 0.75
    near_lock_hours: float = 18.0
    near_lock_min_prob: float = 0.92
    near_lock_max_price: float = 0.82
    near_lock_size_multiplier: float = 1.0
    near_lock_sigma_f: float = 0.75
    near_lock_sigma_c: float = 0.4
    underdispersion_ratio_min: float = 1.6
    underdispersion_tail_max_price: float = 0.14
    underdispersion_size_multiplier: float = 0.35
    model_lag_min_probability_shift: float = 0.08
    model_lag_max_reprice_ratio: float = 0.5
    model_lag_size_multiplier: float = 0.5
    no_trade_min_ev: float = 0.15
    no_trade_size_multiplier: float = 0.35

    @classmethod
    def from_mapping(cls, cfg: dict, *, min_ev: float, max_price: float, max_slippage: float) -> "StrategyConfig":
        return cls(
            min_ev=float(min_ev),
            max_price=float(max_price),
            max_slippage=float(max_slippage),
            max_relative_spread=float(cfg.get("max_relative_spread", 0.25)),
            strategy_calibrated_mean_enabled=bool(cfg.get("strategy_calibrated_mean_enabled", True)),
            strategy_near_lock_enabled=bool(cfg.get("strategy_near_lock_enabled", True)),
            strategy_underdispersion_enabled=bool(cfg.get("strategy_underdispersion_enabled", False)),
            strategy_model_lag_enabled=bool(cfg.get("strategy_model_lag_enabled", True)),
            enable_no_trades=bool(cfg.get("enable_no_trades", False)),
            calibrated_mean_min_samples=int(cfg.get("calibrated_mean_min_samples", cfg.get("calibration_min", 30))),
            standard_price_stop_enabled=bool(cfg.get("standard_price_stop_enabled", False)),
            standard_stop_loss_fraction=float(cfg.get("standard_stop_loss_fraction", 0.20)),
            standard_trailing_activation_return=float(cfg.get("standard_trailing_activation_return", 0.20)),
            standard_take_profit_enabled=bool(cfg.get("standard_take_profit_enabled", True)),
            standard_take_profit_24_48_price=float(cfg.get("standard_take_profit_24_48_price", 0.85)),
            standard_take_profit_48_plus_price=float(cfg.get("standard_take_profit_48_plus_price", 0.75)),
            near_lock_hours=float(cfg.get("near_lock_hours", 18.0)),
            near_lock_min_prob=float(cfg.get("near_lock_min_prob", 0.92)),
            near_lock_max_price=float(cfg.get("near_lock_max_price", 0.82)),
            near_lock_size_multiplier=float(cfg.get("near_lock_size_multiplier", 1.0)),
            near_lock_sigma_f=float(cfg.get("near_lock_sigma_f", 0.75)),
            near_lock_sigma_c=float(cfg.get("near_lock_sigma_c", 0.4)),
            underdispersion_ratio_min=float(cfg.get("underdispersion_ratio_min", 1.6)),
            underdispersion_tail_max_price=float(cfg.get("underdispersion_tail_max_price", 0.14)),
            underdispersion_size_multiplier=float(cfg.get("underdispersion_size_multiplier", 0.35)),
            model_lag_min_probability_shift=float(cfg.get("model_lag_min_probability_shift", 0.08)),
            model_lag_max_reprice_ratio=float(cfg.get("model_lag_max_reprice_ratio", 0.5)),
            model_lag_size_multiplier=float(cfg.get("model_lag_size_multiplier", 0.5)),
            no_trade_min_ev=float(cfg.get("no_trade_min_ev", 0.15)),
            no_trade_size_multiplier=float(cfg.get("no_trade_size_multiplier", 0.35)),
        )


@dataclass(frozen=True)
class BucketQuote:
    market_id: str
    question: str
    bucket_low: float
    bucket_high: float
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float
    volume: float
    fee_rate: float
    yes_token_id: str = ""
    no_token_id: str = ""
    previous_yes_ask: float | None = None
    previous_no_ask: float | None = None
    yes_quote_verified: bool = True
    no_quote_verified: bool = False


@dataclass(frozen=True)
class ForecastContext:
    city_slug: str
    unit: str
    hours_left: float
    horizon: str
    raw_forecast_temp: float
    corrected_forecast_temp: float
    forecast_source: str
    sigma: float
    snapshot_ts: str | None = None
    previous_corrected_forecast_temp: float | None = None
    observed_high_so_far: float | None = None
    observed_high_complete: bool = False
    forecast_remaining_max: float | None = None
    source_spread: float | None = None
    ensemble_spread: float | None = None
    forecast_bias: float = 0.0
    forecast_raw_bias: float = 0.0
    forecast_lead_bucket: str | None = None
    forecast_calibration_n: int = 0
    forecast_calibration_scope: str = "none"


@dataclass(frozen=True)
class StrategyCandidate:
    strategy: str
    side: str
    probability: float
    fair_price: float
    edge: float
    ev: float
    entry_price: float
    bid_price: float
    spread: float
    bucket_low: float
    bucket_high: float
    market_id: str
    question: str
    forecast_source: str
    sigma: float
    fee_rate: float
    size_multiplier: float
    exit_policy: str
    token_id: str = ""
    raw_probability: float | None = None
    raw_ev: float | None = None
    observed_high_so_far: float | None = None
    forecast_remaining_max: float | None = None
    dispersion_ratio: float | None = None
    source_spread: float | None = None
    probability_shift: float | None = None
    market_price_shift: float | None = None
    reason: str = ""
    created_at_ts: float = field(default_factory=time.time)


@dataclass(frozen=True)
class PriceExitDecision:
    """Pure market-price exit decision shared by paper and live execution."""

    reason: str | None = None
    stop_price: float | None = None
    trailing_activated: bool = False


def calibration_gate_reason(context: ForecastContext, config: StrategyConfig) -> str | None:
    required = max(0, int(config.calibrated_mean_min_samples))
    available = max(0, int(context.forecast_calibration_n))
    if available < required:
        return f"calibration {available}/{required}"
    return None


def initial_standard_stop_price(*, entry_bid: float, config: StrategyConfig) -> float | None:
    """Anchor an optional stop to the executable entry bid, never the ask."""
    if not config.standard_price_stop_enabled:
        return None
    fraction = min(max(float(config.standard_stop_loss_fraction), 0.0), 1.0)
    return round(max(0.0, float(entry_bid)) * (1.0 - fraction), 6)


def evaluate_price_exit(
    *,
    entry_price: float,
    current_price: float,
    hours_left: float,
    exit_policy: str,
    stop_price: float | None,
    trailing_activated: bool,
    config: StrategyConfig,
) -> PriceExitDecision:
    """Return an exit decision without performing I/O or mutating a position."""
    if exit_policy == EXIT_HOLD_TO_RESOLUTION:
        return PriceExitDecision(stop_price=stop_price, trailing_activated=trailing_activated)

    if config.standard_take_profit_enabled:
        target = None
        if 24.0 <= hours_left < 48.0:
            target = config.standard_take_profit_24_48_price
        elif hours_left >= 48.0:
            target = config.standard_take_profit_48_plus_price
        if target is not None and current_price >= target:
            return PriceExitDecision(
                reason="take_profit",
                stop_price=stop_price,
                trailing_activated=trailing_activated,
            )

    if not config.standard_price_stop_enabled:
        return PriceExitDecision(stop_price=stop_price, trailing_activated=trailing_activated)

    activation_price = entry_price * (1.0 + config.standard_trailing_activation_return)
    if not trailing_activated and current_price >= activation_price:
        stop_price = round(entry_price, 6)
        trailing_activated = True

    if stop_price is not None and current_price <= stop_price:
        reason = "trailing_stop" if trailing_activated and stop_price >= entry_price else "stop_loss"
        return PriceExitDecision(reason, stop_price, trailing_activated)

    return PriceExitDecision(stop_price=stop_price, trailing_activated=trailing_activated)


def norm_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def bucket_probability(forecast: float, low: float, high: float, sigma: float) -> float:
    sigma = max(float(sigma), 1e-9)
    forecast = float(forecast)
    if low == -999:
        return norm_cdf((high - forecast) / sigma)
    if high == 999:
        return 1.0 - norm_cdf((low - forecast) / sigma)
    if low == high:
        low -= 0.5
        high += 0.5
    return norm_cdf((high - forecast) / sigma) - norm_cdf((low - forecast) / sigma)


def near_lock_probability(
    *,
    observed_high: float,
    forecast_remaining_max: float | None,
    low: float,
    high: float,
    unit: str,
    sigma: float | None = None,
) -> float:
    """Probability for final daily high using late-day observed-high logic."""
    if high != 999 and observed_high > high:
        return 0.0

    final_estimate = max(
        float(observed_high),
        float(forecast_remaining_max) if forecast_remaining_max is not None else float(observed_high),
    )
    near_sigma = sigma if sigma is not None else (0.75 if unit.upper() == "F" else 0.4)
    return bucket_probability(final_estimate, low, high, near_sigma)


def source_spread_from_values(values: list[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if len(numeric) < 2:
        return None
    return max(numeric) - min(numeric)


def dispersion_ratio(*, historical_sigma: float, ensemble_spread: float | None) -> float | None:
    """Compare calibrated residual sigma with a true ensemble standard deviation."""
    if ensemble_spread is None or ensemble_spread <= 0:
        return None
    return float(historical_sigma) / float(ensemble_spread)


def market_is_unrepriced(
    *,
    fair_price_shift: float,
    previous_ask: float | None,
    current_ask: float,
    max_reprice_ratio: float,
) -> bool:
    """True when the ask moved less than the permitted share of fair-value move."""
    if previous_ask is None or abs(fair_price_shift) <= 0:
        return False
    if not 0.0 <= max_reprice_ratio <= 1.0:
        raise ValueError("max_reprice_ratio must be between 0 and 1")
    market_shift = float(current_ask) - float(previous_ask)
    repriced_in_fair_direction = market_shift * (1.0 if fair_price_shift > 0 else -1.0)
    return repriced_in_fair_direction < abs(fair_price_shift) * max_reprice_ratio


def is_in_bucket(value: float, low: float, high: float) -> bool:
    if low == -999:
        return value <= high
    if high == 999:
        return value >= low
    return low <= value <= high


def is_tail_bucket(value: float, low: float, high: float) -> bool:
    return not is_in_bucket(value, low, high)


def quote_for_side(bucket: BucketQuote, side: str) -> tuple[float, float, str]:
    if side == NO:
        return bucket.no_bid, bucket.no_ask, bucket.no_token_id
    return bucket.yes_bid, bucket.yes_ask, bucket.yes_token_id


def quote_is_verified(bucket: BucketQuote, side: str) -> bool:
    return bucket.no_quote_verified if side == NO else bucket.yes_quote_verified


def probability_for_side(yes_probability: float, side: str) -> float:
    yes_probability = max(0.0, min(1.0, float(yes_probability)))
    if side == NO:
        return 1.0 - yes_probability
    return yes_probability


def generate_strategy_candidates(
    *,
    buckets: list[BucketQuote],
    context: ForecastContext,
    config: StrategyConfig,
) -> list[StrategyCandidate]:
    candidates: list[StrategyCandidate] = []
    for bucket in buckets:
        if config.strategy_calibrated_mean_enabled:
            _append_calibrated_mean(candidates, bucket, context, config)
        if config.strategy_near_lock_enabled:
            _append_near_lock(candidates, bucket, context, config)
        if config.strategy_underdispersion_enabled:
            _append_underdispersion_tail(candidates, bucket, context, config)
        if config.strategy_model_lag_enabled:
            _append_model_lag(candidates, bucket, context, config)

    candidates.sort(key=lambda candidate: candidate.ev, reverse=True)
    return candidates


def _append_calibrated_mean(
    candidates: list[StrategyCandidate],
    bucket: BucketQuote,
    context: ForecastContext,
    config: StrategyConfig,
) -> None:
    if calibration_gate_reason(context, config) is not None:
        return
    yes_probability = bucket_probability(
        context.corrected_forecast_temp,
        bucket.bucket_low,
        bucket.bucket_high,
        context.sigma,
    )
    raw_probability = bucket_probability(
        context.raw_forecast_temp,
        bucket.bucket_low,
        bucket.bucket_high,
        context.sigma,
    )
    if is_in_bucket(context.corrected_forecast_temp, bucket.bucket_low, bucket.bucket_high):
        _append_side_candidate(
            candidates,
            strategy=CALIBRATED_MEAN,
            side=YES,
            bucket=bucket,
            context=context,
            yes_probability=yes_probability,
            raw_yes_probability=raw_probability,
            max_price=config.max_price,
            max_spread=config.max_slippage,
            max_relative_spread=config.max_relative_spread,
            min_ev=config.min_ev,
            size_multiplier=1.0,
            exit_policy=EXIT_STANDARD,
            reason="corrected forecast inside bucket",
        )

    if config.enable_no_trades:
        _append_side_candidate(
            candidates,
            strategy=CALIBRATED_MEAN,
            side=NO,
            bucket=bucket,
            context=context,
            yes_probability=yes_probability,
            raw_yes_probability=raw_probability,
            max_price=config.max_price,
            max_spread=config.max_slippage,
            max_relative_spread=config.max_relative_spread,
            min_ev=config.no_trade_min_ev,
            size_multiplier=config.no_trade_size_multiplier,
            exit_policy=EXIT_STANDARD,
            reason="YES bucket overpriced versus calibrated probability",
        )


def _append_near_lock(
    candidates: list[StrategyCandidate],
    bucket: BucketQuote,
    context: ForecastContext,
    config: StrategyConfig,
) -> None:
    if context.hours_left > config.near_lock_hours:
        return
    if not context.observed_high_complete or context.observed_high_so_far is None:
        return
    if context.forecast_remaining_max is None:
        return
    near_sigma = config.near_lock_sigma_f if context.unit.upper() == "F" else config.near_lock_sigma_c
    yes_probability = near_lock_probability(
        observed_high=context.observed_high_so_far,
        forecast_remaining_max=context.forecast_remaining_max,
        low=bucket.bucket_low,
        high=bucket.bucket_high,
        unit=context.unit,
        sigma=near_sigma,
    )
    if yes_probability < config.near_lock_min_prob:
        return
    _append_side_candidate(
        candidates,
        strategy=NEAR_LOCK,
        side=YES,
        bucket=bucket,
        context=context,
        yes_probability=yes_probability,
        raw_yes_probability=None,
        max_price=config.near_lock_max_price,
        max_spread=config.max_slippage,
        max_relative_spread=config.max_relative_spread,
        min_ev=config.min_ev,
        size_multiplier=config.near_lock_size_multiplier,
        exit_policy=EXIT_HOLD_TO_RESOLUTION,
        reason="late-day observed high makes bucket near-locked",
    )


def _append_underdispersion_tail(
    candidates: list[StrategyCandidate],
    bucket: BucketQuote,
    context: ForecastContext,
    config: StrategyConfig,
) -> None:
    if not (24.0 <= context.hours_left <= 72.0):
        return
    ratio = dispersion_ratio(
        historical_sigma=context.sigma,
        ensemble_spread=context.ensemble_spread,
    )
    if ratio is None or ratio < config.underdispersion_ratio_min:
        return
    if not is_tail_bucket(context.corrected_forecast_temp, bucket.bucket_low, bucket.bucket_high):
        return
    yes_probability = bucket_probability(
        context.corrected_forecast_temp,
        bucket.bucket_low,
        bucket.bucket_high,
        context.sigma,
    )
    _append_side_candidate(
        candidates,
        strategy=UNDERDISPERSION_TAIL,
        side=YES,
        bucket=bucket,
        context=context,
        yes_probability=yes_probability,
        raw_yes_probability=None,
        max_price=config.underdispersion_tail_max_price,
        max_spread=config.max_slippage,
        max_relative_spread=config.max_relative_spread,
        min_ev=config.min_ev,
        size_multiplier=config.underdispersion_size_multiplier,
        exit_policy=EXIT_STANDARD,
        dispersion_ratio_value=ratio,
        reason="calibrated residual sigma wider than true ensemble spread",
    )


def _append_model_lag(
    candidates: list[StrategyCandidate],
    bucket: BucketQuote,
    context: ForecastContext,
    config: StrategyConfig,
) -> None:
    if context.previous_corrected_forecast_temp is None:
        return
    previous_probability = bucket_probability(
        context.previous_corrected_forecast_temp,
        bucket.bucket_low,
        bucket.bucket_high,
        context.sigma,
    )
    current_probability = bucket_probability(
        context.corrected_forecast_temp,
        bucket.bucket_low,
        bucket.bucket_high,
        context.sigma,
    )
    shift = current_probability - previous_probability
    if abs(shift) < config.model_lag_min_probability_shift:
        return
    if shift > 0 and market_is_unrepriced(
        fair_price_shift=shift,
        previous_ask=bucket.previous_yes_ask,
        current_ask=bucket.yes_ask,
        max_reprice_ratio=config.model_lag_max_reprice_ratio,
    ):
        _append_side_candidate(
            candidates,
            strategy=MODEL_LAG,
            side=YES,
            bucket=bucket,
            context=context,
            yes_probability=current_probability,
            raw_yes_probability=None,
            max_price=config.max_price,
            max_spread=config.max_slippage,
            max_relative_spread=config.max_relative_spread,
            min_ev=config.min_ev,
            size_multiplier=config.model_lag_size_multiplier,
            exit_policy=EXIT_STANDARD,
            probability_shift=shift,
            market_price_shift=bucket.yes_ask - float(bucket.previous_yes_ask),
            reason="forecast probability rose while YES ask lagged",
        )
    elif shift < 0 and config.enable_no_trades and market_is_unrepriced(
        fair_price_shift=-shift,
        previous_ask=bucket.previous_no_ask,
        current_ask=bucket.no_ask,
        max_reprice_ratio=config.model_lag_max_reprice_ratio,
    ):
        _append_side_candidate(
            candidates,
            strategy=MODEL_LAG,
            side=NO,
            bucket=bucket,
            context=context,
            yes_probability=current_probability,
            raw_yes_probability=None,
            max_price=config.max_price,
            max_spread=config.max_slippage,
            max_relative_spread=config.max_relative_spread,
            min_ev=config.no_trade_min_ev,
            size_multiplier=min(config.model_lag_size_multiplier, config.no_trade_size_multiplier),
            exit_policy=EXIT_STANDARD,
            probability_shift=shift,
            market_price_shift=bucket.no_ask - float(bucket.previous_no_ask),
            reason="forecast probability fell while NO ask lagged",
        )


def _append_side_candidate(
    candidates: list[StrategyCandidate],
    *,
    strategy: str,
    side: str,
    bucket: BucketQuote,
    context: ForecastContext,
    yes_probability: float,
    raw_yes_probability: float | None,
    max_price: float,
    max_spread: float,
    max_relative_spread: float,
    min_ev: float,
    size_multiplier: float,
    exit_policy: str,
    reason: str,
    dispersion_ratio_value: float | None = None,
    probability_shift: float | None = None,
    market_price_shift: float | None = None,
) -> None:
    bid, ask, token_id = quote_for_side(bucket, side)
    if not quote_is_verified(bucket, side):
        return
    spread = ask - bid
    relative_spread = spread / ask if ask > 0 else float("inf")
    if ask <= 0 or ask >= max_price or spread > max_spread or relative_spread > max_relative_spread:
        return
    probability = probability_for_side(yes_probability, side)
    raw_probability = (
        probability_for_side(raw_yes_probability, side)
        if raw_yes_probability is not None
        else None
    )
    ev = fee_adjusted_ev(
        probability=probability,
        price=ask,
        fee_rate=bucket.fee_rate,
    )
    if ev < min_ev:
        return
    raw_ev = (
        fee_adjusted_ev(
            probability=raw_probability,
            price=ask,
            fee_rate=bucket.fee_rate,
        )
        if raw_probability is not None
        else None
    )
    candidates.append(
        StrategyCandidate(
            strategy=strategy,
            side=side,
            probability=round(probability, 6),
            fair_price=round(probability, 6),
            edge=round(probability - ask, 6),
            ev=round(ev, 6),
            entry_price=ask,
            bid_price=bid,
            spread=round(spread, 6),
            bucket_low=bucket.bucket_low,
            bucket_high=bucket.bucket_high,
            market_id=bucket.market_id,
            question=bucket.question,
            forecast_source=context.forecast_source,
            sigma=context.sigma,
            fee_rate=bucket.fee_rate,
            size_multiplier=size_multiplier,
            exit_policy=exit_policy,
            token_id=token_id,
            raw_probability=raw_probability,
            raw_ev=raw_ev,
            observed_high_so_far=context.observed_high_so_far,
            forecast_remaining_max=context.forecast_remaining_max,
            dispersion_ratio=dispersion_ratio_value,
            source_spread=context.source_spread,
            probability_shift=probability_shift,
            market_price_shift=market_price_shift,
            reason=reason,
        )
    )
