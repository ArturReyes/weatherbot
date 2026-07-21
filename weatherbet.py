#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
weatherbet.py — Weather Trading Bot for Polymarket
=====================================================
Tracks weather forecasts from 3 sources (ECMWF, HRRR, METAR),
compares with Polymarket markets, paper trades using Kelly criterion.

Usage:
    python weatherbet.py          # main loop
    python weatherbet.py report   # full report
    python weatherbet.py status   # balance and open positions
"""

import re
import sys
import json
import math
import time
import os
import tempfile
from dotenv import load_dotenv
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from observations import daily_observed_high
from executable_quotes import fetch_executable_quote
from market_resolution import calibration_fields, winning_bucket_from_outcomes
from paper_cohort import (
    build_cohort_archive,
    build_fresh_paper_state,
    cohort_position_records,
    ensure_paper_reset_allowed,
    mark_market_positions_legacy,
)
from validation import PromotionPolicy, chronological_out_of_sample_report, shadow_diagnostics
from weather_http import RetryPolicy, WeatherDataUnavailable, WeatherHttpClient

from calibration import (
    LEAD_TIME_BUCKETS,
    bias_adjusted_forecast,
    calibration_errors,
    decaying_mean_error,
    lead_time_bucket,
    regularized_sigma,
)
from paper_trading import (
    archive_position_for_reentry,
    close_position,
    market_positions,
    paper_reentry_reason,
    record_shadow_signal,
    revalidate_signal_decision,
    settle_shadow_signals,
    settle_paper_market,
    yes_quote,
)
from strategy import (
    EXIT_HOLD_TO_RESOLUTION,
    NO,
    YES,
    BucketQuote,
    ForecastContext,
    StrategyCandidate,
    StrategyConfig,
    calibration_gate_reason,
    evaluate_price_exit,
    generate_strategy_candidates,
    initial_standard_stop_price,
    source_spread_from_values,
)
from trading_risk import (
    RiskLimits,
    assess_trade_risk,
    contract_matches_strategy,
    fee_adjusted_ev,
    fee_adjusted_kelly,
    market_fee_rate,
)

# =============================================================================
# CONFIG
# =============================================================================

load_dotenv(".env.local")
load_dotenv()

CONFIG_FILE = Path("config.json")
with open(CONFIG_FILE, encoding="utf-8") as f:
    _cfg = json.load(f)

BALANCE          = _cfg.get("balance", 10000.0)
MAX_BET          = _cfg.get("max_bet", 20.0)        # max bet per trade
MIN_EV           = _cfg.get("min_ev", 0.10)
MAX_PRICE        = _cfg.get("max_price", 0.45)
MIN_VOLUME       = _cfg.get("min_volume", 500)
MIN_HOURS        = _cfg.get("min_hours", 2.0)
MAX_HOURS        = _cfg.get("max_hours", 72.0)
KELLY_FRACTION   = _cfg.get("kelly_fraction", 0.25)
MAX_SLIPPAGE     = _cfg.get("max_slippage", 0.03)  # max allowed ask-bid spread
MIN_TRADE_NOTIONAL = float(_cfg.get("min_trade_notional", 0.50))
SCAN_INTERVAL    = _cfg.get("scan_interval", 3600)   # every hour
OPPORTUNITY_SCAN_INTERVAL = int(_cfg.get("opportunity_scan_interval_seconds", 300))
ACTIVE_SCAN_INTERVAL = min(SCAN_INTERVAL, OPPORTUNITY_SCAN_INTERVAL)
CALIBRATION_MIN  = _cfg.get("calibration_min", 15)
CALIBRATION_BOOTSTRAP_MIN = int(_cfg.get("calibration_bootstrap_min", 7))
BIAS_DECAY = float(_cfg.get("bias_decay", 0.97))
BIAS_PRIOR_STRENGTH = float(_cfg.get("bias_prior_strength", 20.0))
MAX_BIAS_CORRECTION_F = float(_cfg.get("max_bias_correction_f", 3.0))
MAX_BIAS_CORRECTION_C = float(_cfg.get("max_bias_correction_c", 1.5))
WEATHER_API_USER_AGENT = str(
    os.environ.get("WEATHER_API_USER_AGENT")
    or _cfg.get("weather_api_user_agent", "WeatherBet/1.0 weather-trading-operator")
)
WEATHER_API_RETRY_POLICY = RetryPolicy(
    max_attempts=int(_cfg.get("weather_api_max_attempts", 3)),
    base_delay_seconds=float(_cfg.get("weather_api_retry_base_seconds", 0.75)),
    max_delay_seconds=float(_cfg.get("weather_api_retry_max_seconds", 6.0)),
)
METAR_CACHE_TTL_SECONDS = float(_cfg.get("forecast_cache_ttl_metar_seconds", 45))
# VC_KEY: fetch from env var first, fall back to config.json
VC_KEY           = os.environ.get("VC_KEY") or _cfg.get("vc_key", "")

SIGMA_F = 2.0
SIGMA_C = 1.2
OPEN_METEO_HRRR_MODEL = "ncep_hrrr_conus"

DATA_DIR         = Path("data")
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE       = DATA_DIR / "state.json"
MARKETS_DIR      = DATA_DIR / "markets"
MARKETS_DIR.mkdir(exist_ok=True)
CALIBRATION_FILE = DATA_DIR / "calibration.json"
PAPER_LOG_FILE   = Path(_cfg.get("paper_log_file", "paper_trading.log"))
EVALUATIONS_DIR  = DATA_DIR / "evaluations"
EVALUATIONS_DIR.mkdir(exist_ok=True)
STRATEGY_CONFIG  = StrategyConfig.from_mapping(
    _cfg,
    min_ev=MIN_EV,
    max_price=MAX_PRICE,
    max_slippage=MAX_SLIPPAGE,
)
PAPER_REENTRY_ENABLED = bool(_cfg.get("paper_reentry_enabled", True))
PAPER_REENTRY_COOLDOWN_MINUTES = float(_cfg.get("paper_reentry_cooldown_minutes", 60))
PAPER_MAX_ENTRIES_PER_MARKET = int(_cfg.get("paper_max_entries_per_market", 2))
PAPER_RISK_LIMITS = RiskLimits(
    max_total_exposure_pct=float(_cfg.get("max_total_exposure_pct", 0.25)),
    max_event_exposure_pct=float(_cfg.get("max_event_exposure_pct", 0.10)),
    max_daily_loss_pct=float(_cfg.get("max_daily_loss_pct", 0.05)),
    max_open_positions=int(_cfg.get("max_open_positions", 5)),
    max_signal_age_seconds=float(_cfg.get("max_signal_age_seconds", 120)),
)
PROMOTION_POLICY = PromotionPolicy.from_mapping(_cfg)
PAPER_COHORT_ID = _cfg.get("paper_cohort_id")

_WEATHER_HTTP = WeatherHttpClient(
    user_agent=WEATHER_API_USER_AGENT,
    retry_policy=WEATHER_API_RETRY_POLICY,
)
_METAR_CACHE: dict[str, float] = {}
_METAR_CACHE_FETCHED_AT = 0.0

# =============================================================================
# TERMINAL OUTPUT
# =============================================================================

ANSI_ENABLED = (
    sys.stdout.isatty()
    and not os.environ.get("NO_COLOR")
    and os.environ.get("TERM", "").lower() != "dumb"
)

ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
}

EVENT_COLORS = {
    "BUY": "green",
    "NEAR_LOCK": "green",
    "TAIL": "cyan",
    "MODEL_LAG": "cyan",
    "NO": "yellow",
    "SKIP": "yellow",
    "SHADOW": "cyan",
    "WARN": "yellow",
    "STOP": "red",
    "TRAILING BE": "cyan",
    "CLOSE": "cyan",
    "WIN": "green",
    "LOSS": "red",
    "ERROR": "red",
}

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class TeeStream:
    """Mirror terminal output to a plain-text log file."""

    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = log_file

    def write(self, data):
        self.stream.write(data)
        self.log_file.write(ANSI_ESCAPE_RE.sub("", data))
        return len(data)

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def isatty(self):
        return self.stream.isatty()

    @property
    def encoding(self):
        return getattr(self.stream, "encoding", "utf-8")


def install_paper_logging():
    """Tee paper-run stdout/stderr to PAPER_LOG_FILE without ANSI codes."""
    log_file = PAPER_LOG_FILE.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)
    return original_stdout, original_stderr, log_file


def color(text, name):
    if not ANSI_ENABLED:
        return text
    return f"{ANSI[name]}{text}{ANSI['reset']}"


def badge(label):
    return color(f"[{label}]", EVENT_COLORS.get(label, "bold"))


def scan_event(kind, message):
    return {"kind": kind, "message": message}


def format_bucket(low, high, unit_sym):
    if low == -999:
        return f"≤{high:g}{unit_sym}"
    if high == 999:
        return f"≥{low:g}{unit_sym}"
    if low == high:
        return f"{low:g}{unit_sym}"
    return f"{low:g}-{high:g}{unit_sym}"


def strategy_event_kind(candidate):
    if candidate.side == NO:
        return "NO"
    if candidate.strategy == "near_lock":
        return "NEAR_LOCK"
    if candidate.strategy == "underdispersion_tail":
        return "TAIL"
    if candidate.strategy == "model_lag":
        return "MODEL_LAG"
    return "BUY"


def print_city_result(city_name, events):
    if not events:
        quiet = color("quiet", "dim")
        print(f"  {color('·', 'dim')}     {city_name:<16} {quiet}")
        return

    counts = {}
    for event in events:
        counts[event["kind"]] = counts.get(event["kind"], 0) + 1

    if counts.get("ERROR"):
        status = "ERROR"
    elif counts.get("BUY"):
        status = "BUY"
    elif counts.get("STOP"):
        status = "STOP"
    elif counts.get("CLOSE") or counts.get("TRAILING BE"):
        status = "CLOSE"
    elif counts.get("WARN"):
        status = "WARN"
    elif counts.get("SKIP"):
        status = "SKIP"
    else:
        status = events[0]["kind"]

    summary_parts = []
    for kind, noun in [
        ("BUY", "buy"),
        ("SKIP", "skip"),
        ("STOP", "stop"),
        ("TRAILING BE", "trail"),
        ("CLOSE", "close"),
        ("WARN", "warn"),
        ("ERROR", "error"),
    ]:
        count = counts.get(kind, 0)
        if count:
            summary_parts.append(f"{count} {noun}{'' if count == 1 else 's'}")
    summary = ", ".join(summary_parts)

    print(f"  {badge(status)} {city_name:<16} {summary}")
    for event in events:
        print(f"      └─ {badge(event['kind'])} {event['message']}")


def print_section(title, detail=None):
    line = color("─" * 72, "dim")
    if detail:
        print(f"\n{color(title, 'bold')}  {color(detail, 'dim')}")
    else:
        print(f"\n{color(title, 'bold')}")
    print(line)


LOCATIONS = {
    "nyc":          {"lat": 40.7772,  "lon":  -73.8726, "name": "New York City", "station": "KLGA", "unit": "F", "region": "us"},
    "chicago":      {"lat": 41.9742,  "lon":  -87.9073, "name": "Chicago",       "station": "KORD", "unit": "F", "region": "us"},
    "miami":        {"lat": 25.7959,  "lon":  -80.2870, "name": "Miami",         "station": "KMIA", "unit": "F", "region": "us"},
    "dallas":       {"lat": 32.8471,  "lon":  -96.8518, "name": "Dallas",        "station": "KDAL", "unit": "F", "region": "us"},
    "seattle":      {"lat": 47.4502,  "lon": -122.3088, "name": "Seattle",       "station": "KSEA", "unit": "F", "region": "us"},
    "atlanta":      {"lat": 33.6407,  "lon":  -84.4277, "name": "Atlanta",       "station": "KATL", "unit": "F", "region": "us"},
    "london":       {"lat": 51.5048,  "lon":    0.0495, "name": "London",        "station": "EGLC", "unit": "C", "region": "eu"},
    "paris":        {"lat": 48.9670,  "lon":    2.4280, "name": "Paris",         "station": "LFPB", "unit": "C", "region": "eu"},
    "munich":       {"lat": 48.3537,  "lon":   11.7750, "name": "Munich",        "station": "EDDM", "unit": "C", "region": "eu"},
    "ankara":       {"lat": 40.1281,  "lon":   32.9951, "name": "Ankara",        "station": "LTAC", "unit": "C", "region": "eu"},
    "seoul":        {"lat": 37.4691,  "lon":  126.4505, "name": "Seoul",         "station": "RKSI", "unit": "C", "region": "asia"},
    "tokyo":        {"lat": 35.7647,  "lon":  140.3864, "name": "Tokyo",         "station": "RJTT", "unit": "C", "region": "asia"},
    "shanghai":     {"lat": 31.1443,  "lon":  121.8083, "name": "Shanghai",      "station": "ZSPD", "unit": "C", "region": "asia"},
    "singapore":    {"lat":  1.3502,  "lon":  103.9940, "name": "Singapore",     "station": "WSSS", "unit": "C", "region": "asia"},
    "lucknow":      {"lat": 26.7606,  "lon":   80.8893, "name": "Lucknow",       "station": "VILK", "unit": "C", "region": "asia"},
    "toronto":      {"lat": 43.6772,  "lon":  -79.6306, "name": "Toronto",       "station": "CYYZ", "unit": "C", "region": "ca"},
    "sao-paulo":    {"lat": -23.4356, "lon":  -46.4731, "name": "Sao Paulo",     "station": "SBGR", "unit": "C", "region": "sa"},
    "buenos-aires": {"lat": -34.8222, "lon":  -58.5358, "name": "Buenos Aires",  "station": "SAEZ", "unit": "C", "region": "sa"},
    "wellington":   {"lat": -41.3272, "lon":  174.8052, "name": "Wellington",    "station": "NZWN", "unit": "C", "region": "oc"},
}

TIMEZONES = {
    "nyc": "America/New_York", "chicago": "America/Chicago",
    "miami": "America/New_York", "dallas": "America/Chicago",
    "seattle": "America/Los_Angeles", "atlanta": "America/New_York",
    "london": "Europe/London", "paris": "Europe/Paris",
    "munich": "Europe/Berlin", "ankara": "Europe/Istanbul",
    "seoul": "Asia/Seoul", "tokyo": "Asia/Tokyo",
    "shanghai": "Asia/Shanghai", "singapore": "Asia/Singapore",
    "lucknow": "Asia/Kolkata",
    "toronto": "America/Toronto", "sao-paulo": "America/Sao_Paulo",
    "buenos-aires": "America/Argentina/Buenos_Aires", "wellington": "Pacific/Auckland",
}

MONTHS = ["january","february","march","april","may","june",
          "july","august","september","october","november","december"]

# =============================================================================
# MATH
# =============================================================================

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bucket_prob(forecast, t_low, t_high, sigma=None):
    """Probability that actual temp falls in [t_low, t_high] given forecast uncertainty.

    Uses a Gaussian error model: actual = forecast + N(0, sigma).
    Applies CDF integration for ALL bucket types, not just edge cases.

    Single-value buckets (t_low == t_high) are expanded to ±0.5°F
    because Polymarket resolves to the nearest integer °F at the station.
    """
    s = sigma or SIGMA_F
    if t_low == -999:
        return norm_cdf((t_high - float(forecast)) / s)
    if t_high == 999:
        return 1.0 - norm_cdf((t_low - float(forecast)) / s)
    if t_low == t_high:
        # Single-value bucket → expand to ±0.5°F (station rounding)
        t_low  -= 0.5
        t_high += 0.5
    z_low  = (t_low  - float(forecast)) / s
    z_high = (t_high - float(forecast)) / s
    return norm_cdf(z_high) - norm_cdf(z_low)

def calc_ev(p, price, fee_rate=0.0):
    return fee_adjusted_ev(probability=p, price=price, fee_rate=fee_rate)

def calc_kelly(p, price, fee_rate=0.0):
    full_kelly = fee_adjusted_kelly(
        probability=p,
        price=price,
        fee_rate=fee_rate,
    )
    return round(min(full_kelly * KELLY_FRACTION, 1.0), 4)

def bet_size(kelly, balance):
    raw = kelly * balance
    return round(min(raw, MAX_BET), 2)

# =============================================================================
# CALIBRATION
# =============================================================================

_cal: dict = {}

def install_calibration(cal):
    global _cal
    _cal = cal
    return _cal

def load_cal():
    if CALIBRATION_FILE.exists():
        return install_calibration(json.loads(CALIBRATION_FILE.read_text(encoding="utf-8")))
    return install_calibration({})

def get_sigma(city_slug, source="ecmwf", lead_bucket=None):
    if lead_bucket:
        key = f"{city_slug}_{source}_{lead_bucket}"
        if key in _cal:
            return _cal[key]["sigma"]
    key = f"{city_slug}_{source}"
    if key in _cal:
        return _cal[key]["sigma"]
    return SIGMA_F if LOCATIONS[city_slug]["unit"] == "F" else SIGMA_C

def _max_bias_correction(city_slug):
    return MAX_BIAS_CORRECTION_F if LOCATIONS[city_slug]["unit"] == "F" else MAX_BIAS_CORRECTION_C

def forecast_calibration(city_slug, source, raw_forecast, snapshot_ts, event_end_ts):
    """Return calibrated forecast metadata without fetching or trading."""
    source = source or "ecmwf"
    lead_bucket = lead_time_bucket(snapshot_ts, event_end_ts)
    entry = {}
    calibration_scope = "none"
    if lead_bucket:
        entry = _cal.get(f"{city_slug}_{source}_{lead_bucket}", {})
        if entry:
            calibration_scope = "lead_bucket"
    if not entry:
        entry = _cal.get(f"{city_slug}_{source}", {})
        if entry:
            calibration_scope = "aggregate"

    bias = float(entry.get("bias", 0.0))
    raw_bias = float(entry.get("raw_bias", bias))
    corrected = bias_adjusted_forecast(
        float(raw_forecast),
        bias,
        max_correction=_max_bias_correction(city_slug),
    )
    applied_bias = float(raw_forecast) - corrected
    return {
        "raw_forecast_temp": float(raw_forecast),
        "corrected_forecast_temp": corrected,
        "forecast_temp": corrected,
        "forecast_bias": applied_bias,
        "forecast_raw_bias": raw_bias,
        "forecast_lead_bucket": lead_bucket,
        "forecast_calibration_n": int(entry.get("n", 0)),
        "forecast_calibration_scope": calibration_scope,
        "sigma": get_sigma(city_slug, source, lead_bucket),
    }

def run_calibration(markets, *, reset=False):
    """Recalculates aggregate sigma and lead-bucket bias/sigma calibration."""
    observed = [
        market
        for market in markets
        if market.get("calibration_temp") is not None
        and market.get("calibration_source") == "polymarket_winning_bucket"
        and market.get("city") in LOCATIONS
        and str(market.get("station", "")).upper()
        == str(LOCATIONS[market["city"]]["station"]).upper()
    ]
    compatible_cities = {market["city"] for market in observed}
    cal = {} if reset else load_cal()
    cal = {
        key: value
        for key, value in cal.items()
        if not isinstance(value, dict)
        or (
            value.get("source") != "metar"
            and value.get("city") in LOCATIONS
            and (
                (
                    value.get("station") is not None
                    and str(value["station"]).upper()
                    == str(LOCATIONS[value["city"]]["station"]).upper()
                )
                or (
                    value.get("station") is None
                    and value.get("city") in compatible_cities
                )
            )
        )
    }
    updated = []
    updated_at = datetime.now(timezone.utc).isoformat()
    lead_buckets = [label for _, _, label in LEAD_TIME_BUCKETS] + ["72h_plus"]

    # ``hrrr`` was the old GFS-seamless label.  Real HRRR is stored under the
    # explicit source name below, so historical records cannot contaminate it.
    for source in ["ecmwf", "hrrr_conus"]:
        for city in set(m["city"] for m in observed):
            errors = calibration_errors(observed, city=city, source=source)
            if len(errors) < CALIBRATION_BOOTSTRAP_MIN:
                continue
            key  = f"{city}_{source}"
            prior_sigma = SIGMA_F if LOCATIONS[city]["unit"] == "F" else SIGMA_C
            old  = cal.get(key, {}).get("sigma", prior_sigma)
            floor = 0.5 if LOCATIONS[city]["unit"] == "F" else 0.25
            estimate = decaying_mean_error(
                observed,
                city=city,
                source=source,
                decay=BIAS_DECAY,
                prior_strength=BIAS_PRIOR_STRENGTH,
            )
            new = round(
                regularized_sigma(
                    errors,
                    prior_sigma=prior_sigma,
                    prior_strength=BIAS_PRIOR_STRENGTH,
                    floor=floor,
                ),
                3,
            )
            cal[key] = {
                "city": city,
                "station": LOCATIONS[city]["station"],
                "source": source,
                "lead_bucket": None,
                "bias": round(estimate.bias, 3),
                "raw_bias": round(estimate.raw_bias, 3),
                "sigma": new,
                "n": len(errors),
                "mature": len(errors) >= CALIBRATION_MIN,
                "updated_at": updated_at,
            }
            if abs(new - old) > 0.05:
                updated.append(f"{LOCATIONS[city]['name']} {source}: {old:.2f}->{new:.2f}")

            for lead_bucket in lead_buckets:
                bucket_errors = calibration_errors(
                    observed,
                    city=city,
                    source=source,
                    lead_bucket=lead_bucket,
                )
                if len(bucket_errors) < CALIBRATION_BOOTSTRAP_MIN:
                    continue
                estimate = decaying_mean_error(
                    observed,
                    city=city,
                    source=source,
                    lead_bucket=lead_bucket,
                    decay=BIAS_DECAY,
                    prior_strength=BIAS_PRIOR_STRENGTH,
                )
                bucket_key = f"{city}_{source}_{lead_bucket}"
                cal[bucket_key] = {
                    "city": city,
                    "station": LOCATIONS[city]["station"],
                    "source": source,
                    "lead_bucket": lead_bucket,
                    "bias": round(estimate.bias, 3),
                    "raw_bias": round(estimate.raw_bias, 3),
                    "sigma": round(
                        regularized_sigma(
                            bucket_errors,
                            prior_sigma=prior_sigma,
                            prior_strength=BIAS_PRIOR_STRENGTH,
                            floor=floor,
                        ),
                        3,
                    ),
                    "n": len(bucket_errors),
                    "mature": len(bucket_errors) >= CALIBRATION_MIN,
                    "updated_at": updated_at,
                }

    _atomic_write_text(CALIBRATION_FILE, json.dumps(cal, indent=2))
    install_calibration(cal)
    if updated:
        print(f"  [CAL] {', '.join(updated)}")
    return cal

# =============================================================================
# FORECASTS
# =============================================================================

def get_ecmwf(city_slug, dates):
    """ECMWF via Open-Meteo with bias correction. For all cities."""
    loc = LOCATIONS[city_slug]
    unit = loc["unit"]
    temp_unit = "fahrenheit" if unit == "F" else "celsius"
    result = {}
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={loc['lat']}&longitude={loc['lon']}"
        f"&daily=temperature_2m_max&temperature_unit={temp_unit}"
        f"&forecast_days=7&timezone={TIMEZONES.get(city_slug, 'UTC')}"
        f"&models=ecmwf_ifs025&bias_correction=true"
    )
    try:
        data = _WEATHER_HTTP.get_json(url, provider="Open-Meteo ECMWF", timeout=(5, 10))
        if data is None:
            return result
        if data.get("error"):
            raise WeatherDataUnavailable(f"Open-Meteo ECMWF: {data.get('reason', 'API error')}")
        daily = data.get("daily", {})
        for date, temp in zip(daily.get("time", []), daily.get("temperature_2m_max", [])):
            if date in dates and temp is not None:
                result[date] = round(temp, 1) if unit == "C" else round(temp)
    except (AttributeError, TypeError, ValueError, WeatherDataUnavailable) as e:
        print(f"  [ECMWF] {city_slug}: {e}")
    return result

def get_hrrr(city_slug, dates):
    """Real NOAA HRRR Conus via Open-Meteo, for US cities only."""
    loc = LOCATIONS[city_slug]
    if loc["region"] != "us":
        return {}
    result = {}
    url = (
        f"https://api.open-meteo.com/v1/gfs"
        f"?latitude={loc['lat']}&longitude={loc['lon']}"
        f"&daily=temperature_2m_max&temperature_unit=fahrenheit"
        f"&forecast_days=3&timezone={TIMEZONES.get(city_slug, 'UTC')}"
        f"&models={OPEN_METEO_HRRR_MODEL}"
    )
    try:
        data = _WEATHER_HTTP.get_json(url, provider="Open-Meteo HRRR", timeout=(5, 10))
        if data is None:
            return result
        if data.get("error"):
            raise WeatherDataUnavailable(f"Open-Meteo HRRR: {data.get('reason', 'API error')}")
        daily = data.get("daily", {})
        for date, temp in zip(daily.get("time", []), daily.get("temperature_2m_max", [])):
            if date in dates and temp is not None:
                result[date] = round(temp)
    except (AttributeError, TypeError, ValueError, WeatherDataUnavailable) as e:
        print(f"  [HRRR] {city_slug}: {e}")
    return result

def get_metar_batch(city_slugs=None, *, force=False):
    """Return current METAR temperatures using one multi-station request."""
    global _METAR_CACHE, _METAR_CACHE_FETCHED_AT
    selected = list(city_slugs or LOCATIONS)
    now = time.monotonic()
    cache_is_fresh = now - _METAR_CACHE_FETCHED_AT < METAR_CACHE_TTL_SECONDS
    if not force and cache_is_fresh:
        return {slug: _METAR_CACHE[slug] for slug in selected if slug in _METAR_CACHE}

    station_to_city = {LOCATIONS[slug]["station"]: slug for slug in selected}
    station_ids = ",".join(station_to_city)
    url = f"https://aviationweather.gov/api/data/metar?ids={station_ids}&format=json"
    try:
        data = _WEATHER_HTTP.get_json(url, provider="Aviation Weather METAR", timeout=(5, 10))
        if data is None:
            _METAR_CACHE = {}
            _METAR_CACHE_FETCHED_AT = now
            return {}
        if not isinstance(data, list):
            raise WeatherDataUnavailable("Aviation Weather METAR: unexpected response shape")

        values = {}
        for observation in data:
            if not isinstance(observation, dict):
                continue
            station = str(
                observation.get("icaoId")
                or observation.get("stationId")
                or observation.get("station")
                or ""
            ).upper()
            city_slug = station_to_city.get(station)
            temp_c = observation.get("temp")
            if city_slug is None or temp_c is None:
                continue
            unit = LOCATIONS[city_slug]["unit"]
            value = float(temp_c)
            values[city_slug] = round(value * 9 / 5 + 32) if unit == "F" else round(value, 1)

        _METAR_CACHE = values
        _METAR_CACHE_FETCHED_AT = now
        return dict(values)
    except (TypeError, ValueError, WeatherDataUnavailable) as e:
        _METAR_CACHE = {}
        _METAR_CACHE_FETCHED_AT = now
        print(f"  [METAR] batch: {e}")
        return {}


def get_metar(city_slug):
    """Current observed temperature from a briefly cached station batch."""
    return get_metar_batch().get(city_slug)

def get_remaining_forecast_max(city_slug, date_str):
    """Max remaining hourly forecast temperature for date_str via Open-Meteo."""
    loc = LOCATIONS[city_slug]
    unit = loc["unit"]
    temp_unit = "fahrenheit" if unit == "F" else "celsius"
    tz_name = TIMEZONES.get(city_slug, "UTC")
    try:
        tz = ZoneInfo(tz_name)
    except Exception:
        tz = timezone.utc
    target_date = datetime.strptime(date_str, "%Y-%m-%d").date()
    now_local = datetime.now(tz)
    url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={loc['lat']}&longitude={loc['lon']}"
        f"&hourly=temperature_2m&temperature_unit={temp_unit}"
        f"&forecast_days=2&timezone={tz_name}"
    )
    try:
        data = _WEATHER_HTTP.get_json(url, provider="Open-Meteo hourly", timeout=(5, 10))
        if data is None:
            return None
        if data.get("error"):
            raise WeatherDataUnavailable(f"Open-Meteo hourly: {data.get('reason', 'API error')}")
        hourly = data.get("hourly", {})
        values = []
        for ts, temp in zip(hourly.get("time", []), hourly.get("temperature_2m", [])):
            if temp is None:
                continue
            try:
                observed_at = datetime.fromisoformat(ts)
            except ValueError:
                continue
            if observed_at.tzinfo is None:
                observed_at = observed_at.replace(tzinfo=tz)
            if observed_at.date() == target_date and observed_at >= now_local:
                values.append(float(temp))
        if values:
            return round(max(values), 1) if unit == "C" else round(max(values))
    except (AttributeError, TypeError, ValueError, WeatherDataUnavailable) as e:
        print(f"  [HOURLY] {city_slug} {date_str}: {e}")
    return None

def get_actual_temp(city_slug, date_str):
    """Actual temperature via Visual Crossing for closed markets."""
    loc = LOCATIONS[city_slug]
    station = loc["station"]
    unit = loc["unit"]
    vc_unit = "us" if unit == "F" else "metric"
    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        f"/{station}/{date_str}/{date_str}"
        f"?unitGroup={vc_unit}&key={VC_KEY}&include=days&elements=tempmax"
    )
    try:
        data = requests.get(url, timeout=(5, 8)).json()
        days = data.get("days", [])
        if days and days[0].get("tempmax") is not None:
            return round(float(days[0]["tempmax"]), 1)
    except Exception as e:
        print(f"  [VC] {city_slug} {date_str}: {e}")
    return None

def event_settlement_outcomes(event):
    """Normalize settled YES prices for every temperature bucket in an event."""
    outcomes = []
    for market in event.get("markets", []):
        rng = parse_temp_range(str(market.get("question", "")))
        if rng is None:
            continue
        raw_prices = market.get("outcomePrices", [])
        if isinstance(raw_prices, str):
            try:
                raw_prices = json.loads(raw_prices)
            except (TypeError, json.JSONDecodeError):
                continue
        try:
            yes_price = float(raw_prices[0])
        except (IndexError, TypeError, ValueError):
            continue
        outcomes.append({
            "market_id": str(market.get("id", "")),
            "range": list(rng),
            "settlement_yes_price": yes_price,
        })
    return outcomes

# =============================================================================
# POLYMARKET
# =============================================================================

def get_polymarket_event(city_slug, month, day, year):
    slug = f"highest-temperature-in-{city_slug}-on-{month}-{day}-{year}"
    try:
        r = requests.get(f"https://gamma-api.polymarket.com/events?slug={slug}", timeout=(5, 8))
        data = r.json()
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
    except Exception:
        pass
    return None

def get_market_price(market_id):
    try:
        r = requests.get(f"https://gamma-api.polymarket.com/markets/{market_id}", timeout=(3, 5))
        prices = json.loads(r.json().get("outcomePrices", "[0.5,0.5]"))
        return float(prices[0])
    except Exception:
        return None

def parse_temp_range(question):
    if not question: return None
    num = r'(-?\d+(?:\.\d+)?)'
    if re.search(r'or below', question, re.IGNORECASE):
        m = re.search(num + r'[°]?[FC] or below', question, re.IGNORECASE)
        if m: return (-999.0, float(m.group(1)))
    if re.search(r'or higher', question, re.IGNORECASE):
        m = re.search(num + r'[°]?[FC] or higher', question, re.IGNORECASE)
        if m: return (float(m.group(1)), 999.0)
    m = re.search(r'between ' + num + r'-' + num + r'[°]?[FC]', question, re.IGNORECASE)
    if m: return (float(m.group(1)), float(m.group(2)))
    m = re.search(r'be ' + num + r'[°]?[FC] on', question, re.IGNORECASE)
    if m:
        v = float(m.group(1))
        return (v, v)
    return None

def hours_to_resolution(end_date_str):
    try:
        end = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        return max(0.0, (end - datetime.now(timezone.utc)).total_seconds() / 3600)
    except Exception:
        return 999.0

def in_bucket(forecast, t_low, t_high):
    if t_low == t_high:
        return round(float(forecast)) == round(t_low)
    return t_low <= float(forecast) <= t_high

# =============================================================================
# ATOMIC FILE WRITES
# =============================================================================

def _atomic_write_text(path: Path, content: str):
    """Write text to a file atomically: write to temp, then rename.

    Prevents state corruption if the bot crashes mid-write.
    """
    fd, tmp_path_str = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_path_str, str(path))
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path_str)
        except OSError:
            pass
        raise


# =============================================================================
# MARKET DATA STORAGE
# Each market is stored in a separate file: data/markets/{city}_{date}.json
# =============================================================================

def market_path(city_slug, date_str):
    return MARKETS_DIR / f"{city_slug}_{date_str}.json"

def load_market(city_slug, date_str):
    p = market_path(city_slug, date_str)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return None

def save_market(market):
    p = market_path(market["city"], market["date"])
    _atomic_write_text(p, json.dumps(market, indent=2, ensure_ascii=False))

def load_all_markets():
    markets = []
    for f in MARKETS_DIR.glob("*.json"):
        try:
            markets.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception:
            pass
    return markets

def new_market(city_slug, date_str, event, hours):
    loc = LOCATIONS[city_slug]
    return {
        "city":               city_slug,
        "city_name":          loc["name"],
        "date":               date_str,
        "unit":               loc["unit"],
        "station":            loc["station"],
        "event_end_date":     event.get("endDate", ""),
        "event_id":           str(event.get("id", "")),
        "event_slug":         str(event.get("slug", "")),
        "hours_at_discovery": round(hours, 1),
        "status":             "open",           # open | closed | resolved
        "resolved":           False,
        "position":           None,             # filled when position opens
        "actual_temp":        None,             # filled after resolution
        "resolved_outcome":   None,             # win / loss / no_position
        "pnl":                None,
        "forecast_snapshots": [],               # list of forecast snapshots
        "market_snapshots":   [],               # list of market price snapshots
        "all_outcomes":       [],               # all market buckets
        "shadow_signals":     [],               # diagnostic, never funded
        "created_at":         datetime.now(timezone.utc).isoformat(),
    }


def align_unpositioned_market_station(market, location, *, changed_at):
    """Reset incompatible observations when an untraded market changes station."""
    previous = str(market.get("station", "")).upper()
    current = str(location["station"]).upper()
    if previous == current:
        return True, None
    if market_positions(market) or market.get("shadow_signals"):
        return False, f"station changed {previous or 'unknown'}->{current} with recorded signals"

    market.setdefault("station_migrations", []).append({
        "from": previous or None,
        "to": current,
        "changed_at": changed_at,
        "reason": "resolution_contract_station_changed",
    })
    market["station"] = current
    market["forecast_snapshots"] = []
    market["market_snapshots"] = []
    market["all_outcomes"] = []
    for field in (
        "resolution_contract",
        "settlement_outcomes",
        "winning_market_id",
        "winning_bucket_low",
        "winning_bucket_high",
        "calibration_temp",
        "calibration_bucket_low",
        "calibration_bucket_high",
        "calibration_source",
        "resolution_provider",
        "resolution_station",
        "calibration_validated_at",
        "calibration_exclusion_reason",
    ):
        market.pop(field, None)
    return True, f"station changed {previous or 'unknown'}->{current}; reset snapshots"

# =============================================================================
# STATE (balance and open positions)
# =============================================================================

def load_state():
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    return {
        "balance":          BALANCE,
        "starting_balance": BALANCE,
        "total_trades":     0,
        "wins":             0,
        "losses":           0,
        "peak_balance":     BALANCE,
    }

def save_state(state):
    _atomic_write_text(STATE_FILE, json.dumps(state, indent=2, ensure_ascii=False))


def archive_and_reset_paper():
    """Archive legacy paper results and start a clean bankroll/evaluation cohort."""
    now = datetime.now(timezone.utc)
    started_at = now.isoformat().replace("+00:00", "Z")
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    state = load_state()
    markets = load_all_markets()
    ensure_paper_reset_allowed(markets, state)

    legacy_cohort_id = str(
        state.get("paper_cohort_id")
        or _cfg.get("paper_cohort_id")
        or f"legacy-{stamp}"
    )
    new_cohort_id = f"paper-{stamp}"
    archive_path = EVALUATIONS_DIR / f"paper_cohort_{stamp}.json"
    if archive_path.exists():
        raise ValueError(f"paper archive already exists: {archive_path}")

    archive = build_cohort_archive(
        markets=markets,
        state=state,
        cohort_id=legacy_cohort_id,
        ended_at=started_at,
        evaluation_started_at=_cfg.get("evaluation_started_at"),
    )
    _atomic_write_text(
        archive_path,
        json.dumps(archive, indent=2, ensure_ascii=False),
    )

    for market in markets:
        if mark_market_positions_legacy(market, cohort_id=legacy_cohort_id):
            save_market(market)

    fresh_state = build_fresh_paper_state(
        previous_state=state,
        bankroll=BALANCE,
        cohort_id=new_cohort_id,
        started_at=started_at,
        archive_path=str(archive_path),
    )
    save_state(fresh_state)

    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    config["evaluation_started_at"] = started_at
    config["paper_cohort_id"] = new_cohort_id
    _atomic_write_text(CONFIG_FILE, json.dumps(config, indent=2, ensure_ascii=False) + "\n")

    print(f"\n{'='*55}")
    print("  WEATHERBET — PAPER COHORT RESET")
    print(f"{'='*55}")
    print(f"  Archived:    {archive_path}")
    print(f"  Positions:   {archive['summary']['positions']}")
    print(f"  Legacy PnL:  {archive['summary']['realized_pnl']:+.2f}")
    print(f"  New cohort:  {new_cohort_id}")
    print(f"  Balance:     ${BALANCE:,.2f}")
    print("  Preserved:   market snapshots + calibration")
    print(f"{'='*55}\n")
    return archive_path

# =============================================================================
# CORE LOGIC
# =============================================================================

def take_forecast_snapshot(city_slug, dates):
    """Fetches forecasts from all sources and returns a snapshot."""
    now_str = datetime.now(timezone.utc).isoformat()
    ecmwf   = get_ecmwf(city_slug, dates)
    hrrr    = get_hrrr(city_slug, dates)
    today = datetime.now(ZoneInfo(TIMEZONES.get(city_slug, "UTC"))).strftime("%Y-%m-%d")

    snapshots = {}
    for date in dates:
        snap = {
            "ts":    now_str,
            "ecmwf": ecmwf.get(date),
            "hrrr_conus": hrrr.get(date) if date <= (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%d") else None,
            "metar": get_metar(city_slug) if date == today else None,
        }
        # Best forecast: real HRRR Conus for US D+0/D+1, otherwise ECMWF.
        loc = LOCATIONS[city_slug]
        if loc["region"] == "us" and snap["hrrr_conus"] is not None:
            snap["best"] = snap["hrrr_conus"]
            snap["best_source"] = "hrrr_conus"
        elif snap["ecmwf"] is not None:
            snap["best"] = snap["ecmwf"]
            snap["best_source"] = "ecmwf"
        else:
            snap["best"] = None
            snap["best_source"] = None
        snapshots[date] = snap
    return snapshots

def observed_high_so_far(market, current_metar, city_slug, date_str, now=None):
    """Return daily station high plus a coverage flag for near-lock safety."""
    observations = [
        {"ts": snapshot.get("ts"), "value": snapshot.get("metar")}
        for snapshot in market.get("forecast_snapshots", [])
        if snapshot.get("metar") is not None
    ]
    current_time = now or datetime.now(timezone.utc)
    if current_metar is not None:
        observations.append({"ts": current_time.isoformat(), "value": current_metar})
    return daily_observed_high(
        observations,
        market_date=date_str,
        timezone_name=TIMEZONES.get(city_slug, "UTC"),
        now=current_time,
    )


def previous_corrected_forecast(market):
    for snapshot in reversed(market.get("forecast_snapshots", [])):
        value = snapshot.get("corrected_forecast_temp")
        if value is not None:
            return float(value)
        value = snapshot.get("best")
        if value is not None:
            return float(value)
    return None


def bucket_quotes_from_outcomes(outcomes, previous_quotes=None):
    previous_quotes = previous_quotes or {}
    quotes = []
    for outcome in outcomes:
        yes_bid = float(outcome.get("bid", outcome.get("price", 0.0)))
        yes_ask = float(outcome.get("ask", outcome.get("price", 0.0)))
        quotes.append(
            BucketQuote(
                market_id=str(outcome.get("market_id", "")),
                question=str(outcome.get("question", "")),
                bucket_low=float(outcome["range"][0]),
                bucket_high=float(outcome["range"][1]),
                yes_bid=yes_bid,
                yes_ask=yes_ask,
                no_bid=float(outcome.get("no_bid", max(0.0, 1.0 - yes_ask))),
                no_ask=float(outcome.get("no_ask", min(1.0, 1.0 - yes_bid))),
                volume=float(outcome.get("volume", 0.0)),
                fee_rate=float(outcome.get("fee_rate", 0.0)),
                yes_token_id=str(outcome.get("yes_token_id", "")),
                no_token_id=str(outcome.get("no_token_id", "")),
                previous_yes_ask=previous_quotes.get(str(outcome.get("market_id", "")), {}).get("yes_ask"),
                previous_no_ask=previous_quotes.get(str(outcome.get("market_id", "")), {}).get("no_ask"),
                no_quote_verified=bool(outcome.get("no_quote_verified", False)),
            )
        )
    return quotes


def previous_market_quotes(market):
    """Return the last per-bucket executable quotes captured before this scan."""
    for snapshot in reversed(market.get("market_snapshots", [])):
        quotes = snapshot.get("quotes")
        if isinstance(quotes, dict):
            return quotes
    return {}


def max_price_for_candidate(candidate):
    if candidate.strategy == "near_lock":
        return STRATEGY_CONFIG.near_lock_max_price
    if candidate.strategy == "underdispersion_tail":
        return STRATEGY_CONFIG.underdispersion_tail_max_price
    return STRATEGY_CONFIG.max_price


def min_ev_for_candidate(candidate):
    return STRATEGY_CONFIG.no_trade_min_ev if candidate.side == NO else STRATEGY_CONFIG.min_ev


def current_bid_for_position(position, outcome):
    side = str(position.get("outcome_side") or position.get("side") or YES).upper()
    if side == NO:
        return float(outcome.get("no_bid", max(0.0, 1.0 - float(outcome.get("ask", outcome.get("price", 0.0))))))
    return float(outcome.get("bid", outcome.get("price", 0.0)))


def near_lock_invalidated(position, observed_high):
    if position.get("strategy") != "near_lock" or observed_high is None:
        return False
    if str(position.get("outcome_side", YES)).upper() != YES:
        return False
    bucket_high = float(position.get("bucket_high", 999.0))
    return bucket_high != 999 and float(observed_high) > bucket_high


def candidate_to_position(candidate, *, raw_forecast_temp, forecast_temp, forecast_meta, best_source, opened_at, balance):
    kelly = calc_kelly(candidate.probability, candidate.entry_price, candidate.fee_rate)
    size = round(bet_size(kelly, balance) * candidate.size_multiplier, 2)
    if size <= 0:
        return None
    total_cost_per_share = candidate.entry_price + candidate.fee_rate * candidate.entry_price * (1.0 - candidate.entry_price)
    stop_price = initial_standard_stop_price(
        entry_bid=candidate.bid_price,
        config=STRATEGY_CONFIG,
    )
    return {
        "market_id": candidate.market_id,
        "token_id": candidate.token_id,
        "question": candidate.question,
        "bucket_low": candidate.bucket_low,
        "bucket_high": candidate.bucket_high,
        "entry_price": candidate.entry_price,
        "bid_at_entry": candidate.bid_price,
        "spread": candidate.spread,
        "shares": round(size / total_cost_per_share, 2),
        "cost": size,
        "amount": size,
        "proposed_notional": size,
        "fee_rate": candidate.fee_rate,
        "p": round(candidate.probability, 4),
        "raw_p": round(candidate.raw_probability, 4) if candidate.raw_probability is not None else None,
        "ev": round(candidate.ev, 4),
        "raw_ev": round(candidate.raw_ev, 4) if candidate.raw_ev is not None else None,
        "kelly": round(kelly, 4),
        "forecast_temp": forecast_temp,
        "raw_forecast_temp": raw_forecast_temp,
        "corrected_forecast_temp": forecast_temp,
        "forecast_bias": forecast_meta.get("forecast_bias", 0.0),
        "forecast_raw_bias": forecast_meta.get("forecast_raw_bias", 0.0),
        "forecast_lead_bucket": forecast_meta.get("forecast_lead_bucket"),
        "forecast_calibration_n": forecast_meta.get("forecast_calibration_n", 0),
        "forecast_calibration_scope": forecast_meta.get("forecast_calibration_scope", "none"),
        "forecast_src": best_source,
        "sigma": candidate.sigma,
        "strategy": candidate.strategy,
        "side": candidate.side,
        "outcome_side": candidate.side,
        "fair_price": candidate.fair_price,
        "edge": candidate.edge,
        "observed_high_so_far": candidate.observed_high_so_far,
        "forecast_remaining_max": candidate.forecast_remaining_max,
        "dispersion_ratio": candidate.dispersion_ratio,
        "source_spread": candidate.source_spread,
        "probability_shift": candidate.probability_shift,
        "market_price_shift": candidate.market_price_shift,
        "exit_policy": candidate.exit_policy,
        "stop_reference_price": candidate.bid_price,
        "stop_price": stop_price,
        "trailing_activated": False,
        "strategy_reason": candidate.reason,
        "paper_cohort_id": PAPER_COHORT_ID,
        "opened_at": opened_at,
        "status": "open",
        "pnl": None,
        "exit_price": None,
        "close_reason": None,
        "closed_at": None,
    }

def scan_and_update():
    """Main function of one cycle: updates forecasts, opens/closes positions."""
    global _cal
    now      = datetime.now(timezone.utc)
    state    = load_state()
    balance  = state["balance"]
    new_pos  = 0
    closed   = 0
    resolved = 0

    for city_slug, loc in LOCATIONS.items():
        unit = loc["unit"]
        unit_sym = "F" if unit == "F" else "C"
        city_events = []

        try:
            dates = [(now + timedelta(days=i)).strftime("%Y-%m-%d") for i in range(4)]
            snapshots = take_forecast_snapshot(city_slug, dates)
            time.sleep(0.3)
        except Exception as e:
            city_events.append(scan_event("ERROR", f"forecast snapshot failed: {e}"))
            print_city_result(loc["name"], city_events)
            continue

        for i, date in enumerate(dates):
            dt    = datetime.strptime(date, "%Y-%m-%d")
            event = get_polymarket_event(city_slug, MONTHS[dt.month - 1], dt.day, dt.year)
            if not event:
                continue

            end_date = event.get("endDate", "")
            hours    = hours_to_resolution(end_date) if end_date else 0
            horizon  = f"D+{i}"

            # Load or create market record
            mkt = load_market(city_slug, date)
            if mkt is None:
                if hours < MIN_HOURS or hours > MAX_HOURS:
                    continue
                mkt = new_market(city_slug, date, event, hours)
            elif mkt.get("status") != "resolved":
                station_aligned, station_message = align_unpositioned_market_station(
                    mkt,
                    loc,
                    changed_at=now.isoformat(),
                )
                if not station_aligned:
                    city_events.append(scan_event("ERROR", f"{date} | {station_message}"))
                    save_market(mkt)
                    continue
                if station_message:
                    city_events.append(scan_event("WARN", f"{date} | {station_message}"))

            # Skip if market already resolved
            if mkt["status"] == "resolved":
                continue

            # Update outcomes list — prices taken directly from event
            outcomes = []
            for market in event.get("markets", []):
                question = market.get("question", "")
                mid      = str(market.get("id", ""))
                volume   = float(market.get("volume", 0))
                contract_validation = contract_matches_strategy(
                    market,
                    city_name=loc["name"],
                    station=loc["station"],
                    unit=unit_sym,
                    date_str=date,
                )
                if not contract_validation.valid:
                    continue
                if contract_validation.contract is not None:
                    contract = contract_validation.contract
                    mkt["resolution_contract"] = {
                        "provider": contract.provider,
                        "station": contract.station,
                        "unit": contract.unit,
                        "date": contract.date,
                        "rule": contract.rule,
                    }
                rng      = parse_temp_range(question)
                if not rng:
                    continue
                try:
                    quote = yes_quote(market)
                    bid = quote.bid
                    ask = quote.ask
                    token_ids = market.get("clobTokenIds", "[]")
                    if isinstance(token_ids, str):
                        try:
                            token_ids = json.loads(token_ids)
                        except (TypeError, json.JSONDecodeError):
                            token_ids = []
                    no_quote = (
                        fetch_executable_quote(str(token_ids[1]))
                        if STRATEGY_CONFIG.enable_no_trades and len(token_ids) > 1
                        else None
                    )
                    fee_rate = market_fee_rate(market)
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
                outcomes.append({
                    "question":  question,
                    "market_id": mid,
                    "range":     rng,
                    "bid":       round(bid, 4),
                    "ask":       round(ask, 4),
                    "no_bid":    round(no_quote.bid, 4) if no_quote else 0.0,
                    "no_ask":    round(no_quote.ask, 4) if no_quote else 1.0,
                    "no_quote_verified": no_quote is not None,
                    "yes_token_id": str(token_ids[0]) if len(token_ids) > 0 else "",
                    "no_token_id": str(token_ids[1]) if len(token_ids) > 1 else "",
                    "price":     round(bid, 4),   # for compatibility
                    "spread":    round(ask - bid, 4),
                    "volume":    round(volume, 0),
                    "fee_rate":  fee_rate,
                })

            outcomes.sort(key=lambda x: x["range"][0])
            mkt["all_outcomes"] = outcomes

            # Forecast snapshot
            snap = snapshots.get(date, {})
            prev_forecast_temp = previous_corrected_forecast(mkt)
            raw_forecast_temp = snap.get("best")
            best_source = snap.get("best_source")
            forecast_meta = {}
            forecast_temp = raw_forecast_temp
            if raw_forecast_temp is not None and best_source is not None:
                forecast_meta = forecast_calibration(
                    city_slug,
                    best_source,
                    raw_forecast_temp,
                    snap.get("ts"),
                    end_date,
                )
                forecast_temp = forecast_meta["corrected_forecast_temp"]
            observed = observed_high_so_far(mkt, snap.get("metar"), city_slug, date, now)
            obs_high = observed.high
            remaining_max = (
                get_remaining_forecast_max(city_slug, date)
                if obs_high is not None and hours <= STRATEGY_CONFIG.near_lock_hours
                else None
            )
            source_spread = source_spread_from_values([snap.get("ecmwf"), snap.get("hrrr_conus")])
            forecast_snap = {
                "ts":          snap.get("ts"),
                "horizon":     horizon,
                "hours_left":  round(hours, 1),
                "ecmwf":       snap.get("ecmwf"),
                "hrrr_conus":  snap.get("hrrr_conus"),
                "metar":       snap.get("metar"),
                "best":        snap.get("best"),
                "raw_forecast_temp": raw_forecast_temp,
                "corrected_forecast_temp": forecast_temp,
                "forecast_bias": forecast_meta.get("forecast_bias", 0.0),
                "forecast_raw_bias": forecast_meta.get("forecast_raw_bias", 0.0),
                "forecast_lead_bucket": forecast_meta.get("forecast_lead_bucket"),
                "forecast_calibration_n": forecast_meta.get("forecast_calibration_n", 0),
                "forecast_calibration_scope": forecast_meta.get("forecast_calibration_scope", "none"),
                "observed_high_so_far": obs_high,
                "observed_high_complete": observed.complete,
                "forecast_remaining_max": remaining_max,
                "source_spread": source_spread,
                "best_source": snap.get("best_source"),
            }
            mkt["forecast_snapshots"].append(forecast_snap)

            # Market price snapshot.  Preserve every bucket's ask, not merely
            # the top one, so model-lag can prove an insufficient repricing.
            prior_quotes = previous_market_quotes(mkt)
            top = max(outcomes, key=lambda x: x["price"]) if outcomes else None
            market_snap = {
                "ts":       snap.get("ts"),
                "top_bucket": f"{top['range'][0]}-{top['range'][1]}{unit_sym}" if top else None,
                "top_price":  top["price"] if top else None,
                "quotes": {
                    str(outcome["market_id"]): {
                        "yes_ask": outcome["ask"],
                        "no_ask": outcome["no_ask"],
                    }
                    for outcome in outcomes
                },
            }
            mkt["market_snapshots"].append(market_snap)

            # --- STOP-LOSS AND TRAILING STOP ---
            if mkt.get("position") and mkt["position"].get("status") == "open":
                pos = mkt["position"]
                current_price = None
                for o in outcomes:
                    if o["market_id"] == pos["market_id"]:
                        current_price = o["price"]
                        break

                if current_price is not None:
                    current_price = current_bid_for_position(pos, o)  # sell selected YES/NO token at bid
                    entry = pos["entry_price"]

                    if near_lock_invalidated(pos, obs_high):
                        balance, did_close = close_position(
                            pos,
                            balance=balance,
                            current_price=current_price,
                            reason="near_lock_invalidated",
                            closed_at=snap.get("ts"),
                        )
                        if did_close:
                            closed += 1
                            pnl = pos["pnl"]
                            city_events.append(scan_event(
                                "CLOSE",
                                f"{date} | entry ${entry:.3f} -> exit ${current_price:.3f} | PnL {'+' if pnl >= 0 else ''}{pnl:.2f}",
                            ))
                    else:
                        decision = evaluate_price_exit(
                            entry_price=entry,
                            current_price=current_price,
                            hours_left=hours,
                            exit_policy=pos.get("exit_policy", "standard"),
                            stop_price=pos.get("stop_price"),
                            trailing_activated=bool(pos.get("trailing_activated", False)),
                            config=STRATEGY_CONFIG,
                        )
                        pos["stop_price"] = decision.stop_price
                        pos["trailing_activated"] = decision.trailing_activated
                        if decision.reason is not None:
                            balance, did_close = close_position(
                                pos,
                                balance=balance,
                                current_price=current_price,
                                reason=decision.reason,
                                closed_at=snap.get("ts"),
                            )
                            if did_close:
                                closed += 1
                                reason = {
                                    "stop_loss": "STOP",
                                    "trailing_stop": "TRAILING",
                                    "take_profit": "TAKE",
                                }.get(decision.reason, "CLOSE")
                                pnl = pos["pnl"]
                                city_events.append(scan_event(
                                    reason,
                                    f"{date} | entry ${entry:.3f} -> exit ${current_price:.3f} | PnL {'+' if pnl >= 0 else ''}{pnl:.2f}",
                                ))

            # --- CLOSE POSITION if forecast shifted 2+ degrees ---
            if (mkt.get("position")
                    and mkt["position"].get("status") == "open"
                    and mkt["position"].get("exit_policy") != EXIT_HOLD_TO_RESOLUTION
                    and forecast_temp is not None):
                pos = mkt["position"]
                old_bucket_low  = pos["bucket_low"]
                old_bucket_high = pos["bucket_high"]
                # 2-degree buffer — avoid closing on small forecast fluctuations
                unit = loc["unit"]
                buffer = 2.0 if unit == "F" else 1.0
                mid_bucket = (old_bucket_low + old_bucket_high) / 2 if old_bucket_low != -999 and old_bucket_high != 999 else forecast_temp
                forecast_far = abs(forecast_temp - mid_bucket) > (abs(mid_bucket - old_bucket_low) + buffer)
                if not in_bucket(forecast_temp, old_bucket_low, old_bucket_high) and forecast_far:
                    current_price = None
                    for o in outcomes:
                        if o["market_id"] == pos["market_id"]:
                            current_price = current_bid_for_position(pos, o)
                            break
                    if current_price is not None:
                        balance, did_close = close_position(
                            pos,
                            balance=balance,
                            current_price=current_price,
                            reason="forecast_changed",
                            closed_at=snap.get("ts"),
                        )
                        if did_close:
                            closed += 1
                            pnl = pos["pnl"]
                            city_events.append(scan_event(
                                "CLOSE",
                                f"{date} | forecast changed | PnL {'+' if pnl >= 0 else ''}{pnl:.2f}",
                            ))

            # --- OPEN POSITION ---
            current_position = mkt.get("position")
            may_consider_entry = not current_position or current_position.get("status") != "open"
            if may_consider_entry and forecast_temp is not None and hours >= MIN_HOURS:
                sigma = float(forecast_meta.get("sigma", get_sigma(city_slug, best_source or "ecmwf")))
                buckets = [
                    bucket
                    for bucket in bucket_quotes_from_outcomes(outcomes, prior_quotes)
                    if bucket.volume >= MIN_VOLUME
                ]
                context = ForecastContext(
                    city_slug=city_slug,
                    unit=unit_sym,
                    hours_left=hours,
                    horizon=horizon,
                    raw_forecast_temp=float(raw_forecast_temp),
                    corrected_forecast_temp=float(forecast_temp),
                    forecast_source=best_source or "ecmwf",
                    sigma=sigma,
                    snapshot_ts=snap.get("ts"),
                    previous_corrected_forecast_temp=prev_forecast_temp,
                    observed_high_so_far=obs_high,
                    observed_high_complete=observed.complete,
                    forecast_remaining_max=remaining_max,
                    source_spread=source_spread,
                    forecast_bias=forecast_meta.get("forecast_bias", 0.0),
                    forecast_raw_bias=forecast_meta.get("forecast_raw_bias", 0.0),
                    forecast_lead_bucket=forecast_meta.get("forecast_lead_bucket"),
                    forecast_calibration_n=forecast_meta.get("forecast_calibration_n", 0),
                    forecast_calibration_scope=forecast_meta.get("forecast_calibration_scope", "none"),
                )
                candidates = generate_strategy_candidates(
                    buckets=buckets,
                    context=context,
                    config=STRATEGY_CONFIG,
                )
                gate_reason = calibration_gate_reason(context, STRATEGY_CONFIG)
                if not candidates and gate_reason is not None:
                    city_events.append(scan_event("WAIT", f"{date} | {gate_reason}"))
                best_candidate = candidates[0] if candidates else None
                best_signal = (
                    candidate_to_position(
                        best_candidate,
                        raw_forecast_temp=raw_forecast_temp,
                        forecast_temp=forecast_temp,
                        forecast_meta=forecast_meta,
                        best_source=best_source,
                        opened_at=snap.get("ts"),
                        balance=balance,
                    )
                    if best_candidate is not None
                    else None
                )
                if best_candidate is not None and best_signal is None:
                    city_events.append(scan_event("SKIP", f"{date} | proposed size is zero"))

                if best_signal and best_candidate is not None:
                    # Fetch real bestAsk from Polymarket API for accurate entry price
                    skip_position = False
                    try:
                        r = requests.get(f"https://gamma-api.polymarket.com/markets/{best_signal['market_id']}", timeout=(3, 5))
                        mdata = r.json()
                        decision = revalidate_signal_decision(
                            best_signal,
                            mdata,
                            min_ev=min_ev_for_candidate(best_candidate),
                            max_price=max_price_for_candidate(best_candidate),
                            max_spread=MAX_SLIPPAGE,
                            max_relative_spread=STRATEGY_CONFIG.max_relative_spread,
                            min_trade_notional=MIN_TRADE_NOTIONAL,
                        )
                        if not decision.accepted:
                            if decision.reason == "below_order_minimum" and decision.signal is not None:
                                shadow_added = record_shadow_signal(
                                    mkt,
                                    decision.signal,
                                    recorded_at=snap.get("ts") or now.isoformat(),
                                    skip_reason=decision.reason,
                                )
                                eligibility = decision.eligibility
                                required = eligibility.required_notional if eligibility else None
                                minimum = eligibility.min_order_size if eligibility else None
                                message = (
                                    f"{date} | proposed ${decision.signal['proposed_notional']:.2f} < "
                                    f"required ${required:.2f} ({minimum:g} shares at "
                                    f"${decision.signal['entry_price']:.3f})"
                                )
                                city_events.append(scan_event(
                                    "SHADOW" if shadow_added else "SKIP",
                                    f"[SIZE] {message}",
                                ))
                            else:
                                city_events.append(scan_event(
                                    "SKIP",
                                    f"{date} | revalidation: {decision.reason or 'rejected'}",
                                ))
                            skip_position = True
                        else:
                            best_signal = decision.signal
                    except Exception as e:
                        city_events.append(scan_event(
                            "WARN",
                            f"could not fetch real ask for {best_signal['market_id']}: {e}",
                        ))
                        skip_position = True

                    if not skip_position and best_signal["entry_price"] < max_price_for_candidate(best_candidate):
                        reentry_block = paper_reentry_reason(
                            mkt,
                            now=now,
                            enabled=PAPER_REENTRY_ENABLED,
                            cooldown_minutes=PAPER_REENTRY_COOLDOWN_MINUTES,
                            max_entries=PAPER_MAX_ENTRIES_PER_MARKET,
                        )
                        if reentry_block is not None:
                            city_events.append(scan_event("SKIP", f"{date} | {reentry_block}"))
                            save_market(mkt)
                            continue
                        paper_positions = [
                            position
                            for market in load_all_markets()
                            for position in market_positions(market)
                        ]
                        paper_risk = assess_trade_risk(
                            {"positions": paper_positions},
                            size_usdc=float(best_signal["cost"]),
                            city_slug=city_slug,
                            date_str=date,
                            signal_created_at=best_candidate.created_at_ts,
                            bankroll=float(state.get("starting_balance", BALANCE)),
                            limits=PAPER_RISK_LIMITS,
                            now_ts=time.time(),
                        )
                        if not paper_risk.allowed:
                            city_events.append(scan_event(
                                "SKIP",
                                f"{date} | paper risk gate: {paper_risk.reason}",
                            ))
                        else:
                            if mkt.get("position") is not None:
                                archive_position_for_reentry(mkt)
                            balance -= best_signal["cost"]
                            mkt["position"] = best_signal
                            state["total_trades"] += 1
                            new_pos += 1
                            bucket_label = format_bucket(
                                best_signal["bucket_low"],
                                best_signal["bucket_high"],
                                unit_sym,
                            )
                            event_kind = strategy_event_kind(best_candidate)
                            city_events.append(scan_event(
                                event_kind,
                                f"{best_signal.get('side', YES)} {horizon} {date} | {bucket_label} | "
                                f"{best_signal['strategy']} | ask ${best_signal['entry_price']:.3f} | "
                                f"EV {best_signal['ev']:+.2f} | size ${best_signal['cost']:.2f} | "
                                f"{best_signal['forecast_src'].upper()}",
                            ))

            # Market closed by time
            if hours < 0.5 and mkt["status"] == "open":
                mkt["status"] = "closed"

            save_market(mkt)
            time.sleep(0.1)

        print_city_result(loc["name"], city_events)

    # --- AUTO-RESOLUTION ---
    resolution_events = []
    for mkt in load_all_markets():
        if mkt.get("city") not in LOCATIONS:
            continue
        if mkt.get("actual_temp") is None and mkt.get("status") in {"closed", "resolved"} and VC_KEY:
            actual_temp = get_actual_temp(mkt["city"], mkt["date"])
            if actual_temp is not None:
                mkt["actual_temp"] = actual_temp
                mkt["actual_temp_observed_at"] = now.isoformat()
                save_market(mkt)

        if mkt["status"] == "resolved":
            continue

        try:
            market_date = datetime.strptime(mkt["date"], "%Y-%m-%d")
        except (KeyError, TypeError, ValueError):
            continue
        if market_date.date() > now.date():
            continue
        event = get_polymarket_event(
            mkt["city"],
            MONTHS[market_date.month - 1],
            market_date.day,
            market_date.year,
        )
        if not event:
            continue
        settlement_outcomes = event_settlement_outcomes(event)
        winner_decision = winning_bucket_from_outcomes(settlement_outcomes)
        if winner_decision.bucket is None:
            continue
        winner = winner_decision.bucket
        mkt["settlement_outcomes"] = settlement_outcomes
        mkt["winning_market_id"] = winner.market_id
        mkt["winning_bucket_low"] = winner.low
        mkt["winning_bucket_high"] = winner.high
        contract = mkt.get("resolution_contract") or {}
        if winner.bounded:
            mkt.update(calibration_fields(
                winner,
                provider=str(contract.get("provider") or "polymarket_legacy"),
                station=str(contract.get("station") or mkt.get("station") or "unknown"),
                validated_at=now.isoformat(),
            ))
        else:
            mkt["calibration_exclusion_reason"] = "open_ended_winning_bucket"

        shadow_count = settle_shadow_signals(
            mkt,
            winning_market_id=winner.market_id,
            resolved_at=now.isoformat(),
        )
        positions = market_positions(mkt)
        if not positions:
            mkt["status"] = "resolved"
            mkt["resolved"] = True
            mkt["resolved_at"] = now.isoformat()
            mkt["resolved_outcome"] = "no_position"
            save_market(mkt)
            resolved += 1
            if shadow_count:
                resolution_events.append(scan_event(
                    "SHADOW",
                    f"{mkt['city_name']} {mkt['date']} | resolved {shadow_count} diagnostic signal(s)",
                ))
            continue
        pos = mkt.get("position") or positions[-1]
        transition = settle_paper_market(
            mkt,
            winning_market_id=winner.market_id,
            balance=balance,
            resolved_at=now.isoformat(),
            actual_temp=mkt.get("actual_temp"),
        )
        if not transition.newly_resolved:
            continue
        balance = transition.balance
        for recorded_win in transition.recorded_results:
            if recorded_win:
                state["wins"] += 1
            else:
                state["losses"] += 1

        result = "WIN" if transition.position_won else "LOSS"
        pnl = float(pos.get("pnl", 0.0))
        lifecycle = "held" if transition.position_was_open else f"exited {pos.get('close_reason', 'early')}"
        resolution_events.append(scan_event(
            result,
            f"{mkt['city_name']} {mkt['date']} | eventual {result} | {lifecycle} | "
            f"trade PnL {'+' if pnl >= 0 else ''}{pnl:.2f}",
        ))
        resolved += 1

        save_market(mkt)
        time.sleep(0.3)

    if resolution_events:
        print(f"\n  {color('Resolutions', 'bold')}")
        for event in resolution_events:
            print(f"      └─ {badge(event['kind'])} {event['message']}")

    state["balance"]      = round(balance, 2)
    state["peak_balance"] = max(state.get("peak_balance", balance), balance)
    save_state(state)

    # Run calibration if enough data collected
    all_mkts = load_all_markets()
    calibration_sample_count = len([
        market
        for market in all_mkts
        if market.get("calibration_temp") is not None
        and market.get("calibration_source") == "polymarket_winning_bucket"
    ])
    if calibration_sample_count >= CALIBRATION_BOOTSTRAP_MIN:
        global _cal
        _cal = run_calibration(all_mkts)

    return new_pos, closed, resolved

# =============================================================================
# REPORT
# =============================================================================

def _paper_position_mark_price(market, position):
    """Return the latest executable bid used to mark an open paper position."""
    market_id = position.get("market_id")
    for outcome in market.get("all_outcomes", []):
        if outcome.get("market_id") != market_id:
            continue
        for field in ("bid", "price"):
            value = outcome.get(field)
            if value is None:
                continue
            try:
                mark = float(value)
            except (TypeError, ValueError):
                continue
            if 0.0 <= mark <= 1.0:
                return mark
    return None


def print_shadow_diagnostics(markets, *, heading="Shadow diagnostics"):
    """Display non-funded signals without mixing them into trading results."""
    metrics = shadow_diagnostics(markets)
    print(f"\n  {heading} (not traded; excluded from promotion):")
    if not metrics:
        print("    No shadow signals yet.")
        return metrics
    for strategy, item in metrics.items():
        brier = f"{item.brier_score:.3f}" if item.brier_score is not None else "n/a"
        print(
            f"    {strategy:<22} {item.signals:>3} signals | "
            f"resolved {item.resolved:>3} | wins {item.wins:>3} | Brier {brier}"
        )
    return metrics


def print_status():
    state    = load_state()
    markets  = load_all_markets()
    cohort_records = cohort_position_records(
        markets,
        started_at=PROMOTION_POLICY.evaluation_started_at,
    )
    open_pos = [(market, position) for market, position in cohort_records if position.get("status") == "open"]
    resolved = [position for _, position in cohort_records if position.get("eventual_outcome") is not None]

    bal     = state["balance"]
    start   = state["starting_balance"]
    wins    = state["wins"]
    losses  = state["losses"]
    entries = int(state.get("total_trades", 0))
    results = wins + losses
    marked_positions = [
        (market, position, _paper_position_mark_price(market, position))
        for market, position in open_pos
    ]
    deployed = sum(float(position.get("cost", position.get("amount", 0.0))) for _, position, _ in marked_positions)
    position_value = sum(
        float(position.get("shares", 0.0)) * mark
        for _, position, mark in marked_positions
        if mark is not None
    )
    missing_marks = sum(mark is None for _, _, mark in marked_positions)
    equity = bal + position_value if missing_marks == 0 else None
    ret_pct = ((equity - start) / start * 100) if equity is not None and start else None

    print(f"\n{'='*55}")
    print(f"  WEATHERBET — STATUS")
    print(f"{'='*55}")
    print(f"  Cash:        ${bal:,.2f}")
    print(f"  Deployed:    ${deployed:,.2f}")
    if equity is not None:
        print(f"  Position:    ${position_value:,.2f}  (marked at executable bid)")
        print(
            f"  Equity:      ${equity:,.2f}  "
            f"(start ${start:,.2f}, {'+' if ret_pct >= 0 else ''}{ret_pct:.1f}%)"
        )
    else:
        print(f"  Position:    unavailable ({missing_marks} missing bid mark{'s' if missing_marks != 1 else ''})")
        print(f"  Equity:      unavailable")
    if entries or results:
        result_rate = f" | WR: {wins/results:.0%}" if results else ""
        print(f"  Entries:     {entries} | Results W: {wins} L: {losses}{result_rate}")
    else:
        print("  No trades yet")
    print(f"  Open:        {len(open_pos)}")
    print(f"  Resolved:    {len(resolved)}")

    if open_pos:
        print(f"\n  Open positions:")
        total_unrealized = 0.0
        priced_positions = 0
        for m, pos, current_price in marked_positions:
            unit_sym = "F" if m["unit"] == "F" else "C"
            label    = f"{pos['bucket_low']}-{pos['bucket_high']}{unit_sym}"
            if current_price is None:
                print(f"    {m['city_name']:<16} {m['date']} | {label:<14} | "
                      f"entry ${pos['entry_price']:.3f} -> bid unavailable | "
                      f"uPnL: n/a | {pos['forecast_src'].upper()}")
                continue

            mark_value = current_price * float(pos.get("shares", 0.0))
            cost = float(pos.get("cost", pos.get("amount", 0.0)))
            unrealized = round(mark_value - cost, 2)
            total_unrealized += unrealized
            priced_positions += 1
            pnl_str = f"{'+'if unrealized>=0 else ''}{unrealized:.2f}"

            print(f"    {m['city_name']:<16} {m['date']} | {label:<14} | "
                  f"entry ${pos['entry_price']:.3f} -> bid ${current_price:.3f} | "
                  f"uPnL: {pnl_str} | {pos['forecast_src'].upper()}")

        if priced_positions:
            sign = "+" if total_unrealized >= 0 else ""
            print(f"\n  Unrealized PnL: {sign}{total_unrealized:.2f} (before exit fees)")

    print_shadow_diagnostics(markets)

    print(f"{'='*55}\n")

def print_report():
    markets  = load_all_markets()
    resolved = [m for m in markets if m["status"] == "resolved" and m.get("pnl") is not None]

    print(f"\n{'='*55}")
    print(f"  WEATHERBET — FULL REPORT")
    print(f"{'='*55}")

    if not resolved:
        print("  No resolved markets yet.")
        print_shadow_diagnostics(markets)
        print(f"{'='*55}\n")
        return

    total_pnl = sum(m["pnl"] for m in resolved)
    wins      = [m for m in resolved if m["resolved_outcome"] == "win"]
    losses    = [m for m in resolved if m["resolved_outcome"] == "loss"]

    print(f"\n  Total resolved: {len(resolved)}")
    print(f"  Wins:           {len(wins)} | Losses: {len(losses)}")
    print(f"  Win rate:       {len(wins)/len(resolved):.0%}")
    print(f"  Total PnL:      {'+'if total_pnl>=0 else ''}{total_pnl:.2f}")

    print(f"\n  By city:")
    for city in sorted(set(m["city"] for m in resolved)):
        group = [m for m in resolved if m["city"] == city]
        w     = len([m for m in group if m["resolved_outcome"] == "win"])
        pnl   = sum(m["pnl"] for m in group)
        name  = LOCATIONS[city]["name"]
        print(f"    {name:<16} {w}/{len(group)} ({w/len(group):.0%})  PnL: {'+'if pnl>=0 else ''}{pnl:.2f}")

    print(f"\n  Market details:")
    for m in sorted(resolved, key=lambda x: x["date"]):
        pos      = m.get("position", {})
        unit_sym = "F" if m["unit"] == "F" else "C"
        snaps    = m.get("forecast_snapshots", [])
        first_fc = snaps[0]["best"] if snaps else None
        last_fc  = snaps[-1]["best"] if snaps else None
        label    = f"{pos.get('bucket_low')}-{pos.get('bucket_high')}{unit_sym}" if pos else "no position"
        result   = m["resolved_outcome"].upper()
        pnl_str  = f"{'+'if m['pnl']>=0 else ''}{m['pnl']:.2f}" if m["pnl"] is not None else "-"
        fc_str   = f"forecast {first_fc}->{last_fc}{unit_sym}" if first_fc else "no forecast"
        actual   = f"actual {m['actual_temp']}{unit_sym}" if m["actual_temp"] else ""
        print(f"    {m['city_name']:<16} {m['date']} | {label:<14} | {fc_str} | {actual} | {result} {pnl_str}")

    print(f"{'='*55}\n")
    print_validation_report()


def live_readiness_report():
    """Return the chronological paper-trading promotion decision for live use."""
    state = load_state()
    return chronological_out_of_sample_report(
        load_all_markets(),
        policy=PROMOTION_POLICY,
        bankroll=float(state.get("starting_balance", BALANCE)),
        required_strategies=active_strategy_names(),
    )


def active_strategy_names():
    strategies = []
    if STRATEGY_CONFIG.strategy_calibrated_mean_enabled:
        strategies.append("calibrated_mean")
    if STRATEGY_CONFIG.strategy_near_lock_enabled:
        strategies.append("near_lock")
    if STRATEGY_CONFIG.strategy_underdispersion_enabled:
        strategies.append("underdispersion_tail")
    if STRATEGY_CONFIG.strategy_model_lag_enabled:
        strategies.append("model_lag")
    return tuple(strategies)


def print_validation_report():
    report = live_readiness_report()
    print(f"\n{'='*55}")
    print("  WEATHERBET — LIVE READINESS")
    print(f"{'='*55}")
    print(f"  Holdout:     latest {report.holdout_fraction:.0%} of closed paper trades")
    if report.evaluation_started_at:
        print(f"  Cohort:      entries opened on/after {report.evaluation_started_at}")
    if not report.holdout:
        print("  No closed paper trades yet.")
    for strategy, metrics in report.holdout.items():
        roi = f"{metrics.realized_roi:+.1%}" if metrics.realized_roi is not None else "n/a"
        brier = f"{metrics.brier_score:.3f}" if metrics.brier_score is not None else "n/a"
        print(
            f"  {strategy:<22} {metrics.trades:>3} trades | ROI {roi:>7} | "
            f"Brier {brier:>5} ({metrics.brier_samples}) | DD {metrics.max_drawdown_pct:.1%}"
        )
    if report.decision.ready:
        print("\n  Promotion:  READY — holdout gates passed")
    else:
        print("\n  Promotion:  NOT READY")
        for reason in report.decision.reasons:
            print(f"    - {reason}")
    print_shadow_diagnostics(load_all_markets())
    print("  Note: paper fills are assumed; post-entry price movement is not reported as CLV.")
    print(f"{'='*55}\n")
    return report


def _contract_rounded_temperature(value):
    numeric = float(value)
    return math.floor(numeric + 0.5) if numeric >= 0 else math.ceil(numeric - 0.5)


def repair_calibration(*, apply_changes):
    """Rebuild calibration provenance from locally stored Polymarket winners."""
    markets = load_all_markets()
    validated_at = datetime.now(timezone.utc).isoformat()
    report = {
        "mode": "apply" if apply_changes else "dry_run",
        "created_at": validated_at,
        "scanned": len(markets),
        "recoverable": 0,
        "tail_skipped": 0,
        "unresolved": 0,
        "ambiguous": 0,
        "invalid": 0,
        "provider_disagreements": 0,
        "annotated": 0,
    }
    repaired = []
    for market in markets:
        outcomes = market.get("settlement_outcomes") or market.get("all_outcomes") or []
        decision = winning_bucket_from_outcomes(outcomes, allow_legacy_price=True)
        if decision.bucket is None:
            if decision.reason == "unresolved":
                report["unresolved"] += 1
            elif decision.reason == "ambiguous_winner":
                report["ambiguous"] += 1
            else:
                report["invalid"] += 1
            continue
        winner = decision.bucket
        if not winner.bounded:
            report["tail_skipped"] += 1
            if apply_changes:
                market["winning_market_id"] = winner.market_id
                market["winning_bucket_low"] = winner.low
                market["winning_bucket_high"] = winner.high
                market["calibration_exclusion_reason"] = "open_ended_winning_bucket"
                repaired.append(market)
            continue

        report["recoverable"] += 1
        actual = market.get("actual_temp")
        if actual is not None:
            try:
                rounded_actual = _contract_rounded_temperature(actual)
                if not winner.low <= rounded_actual <= winner.high:
                    report["provider_disagreements"] += 1
            except (TypeError, ValueError):
                report["provider_disagreements"] += 1
        if not apply_changes:
            continue

        contract = market.get("resolution_contract") or {}
        market["winning_market_id"] = winner.market_id
        market["winning_bucket_low"] = winner.low
        market["winning_bucket_high"] = winner.high
        market.update(calibration_fields(
            winner,
            provider=str(contract.get("provider") or "polymarket_legacy"),
            station=str(contract.get("station") or market.get("station") or "unknown"),
            validated_at=validated_at,
        ))
        repaired.append(market)

    if apply_changes:
        archive_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        if CALIBRATION_FILE.exists():
            backup = EVALUATIONS_DIR / f"calibration_before_repair_{archive_stamp}.json"
            _atomic_write_text(backup, CALIBRATION_FILE.read_text(encoding="utf-8"))
        for market in repaired:
            save_market(market)
        report["annotated"] = len(repaired)
        run_calibration(load_all_markets(), reset=True)
        report_path = EVALUATIONS_DIR / f"calibration_repair_{archive_stamp}.json"
        _atomic_write_text(report_path, json.dumps(report, indent=2))
        report["report_path"] = str(report_path)

    print(f"\n{'='*55}")
    print("  WEATHERBET — CALIBRATION REPAIR")
    print(f"{'='*55}")
    print(f"  Mode:          {report['mode']}")
    print(f"  Scanned:       {report['scanned']}")
    print(f"  Recoverable:   {report['recoverable']}")
    print(f"  Tail skipped:  {report['tail_skipped']}")
    print(f"  Unresolved:    {report['unresolved']}")
    print(f"  Ambiguous:     {report['ambiguous']}")
    print(f"  Invalid:       {report['invalid']}")
    print(f"  VC mismatch:   {report['provider_disagreements']}")
    if apply_changes:
        print(f"  Annotated:     {report['annotated']}")
        print(f"  Audit:         {report['report_path']}")
    else:
        print("  No files changed. Use --apply after reviewing these counts.")
    print(f"{'='*55}\n")
    return report

# =============================================================================
# MAIN LOOP
# =============================================================================

MONITOR_INTERVAL = 600  # monitor positions every 10 minutes

def monitor_positions():
    """Quick stop check on open positions without full scan."""
    markets  = load_all_markets()
    open_pos = [m for m in markets if m.get("position") and m["position"].get("status") == "open"]
    if not open_pos:
        return 0

    state   = load_state()
    balance = state["balance"]
    closed  = 0

    for mkt in open_pos:
        pos = mkt["position"]
        mid = pos["market_id"]

        # Fetch the selected YES/NO token's executable bid.
        current_price = None
        try:
            quote = fetch_executable_quote(str(pos.get("token_id", "")))
            if quote is not None:
                current_price = float(quote.bid)
        except Exception:
            pass

        # Fallback to cached price if API failed
        if current_price is None:
            for o in mkt.get("all_outcomes", []):
                if o["market_id"] == mid:
                    current_price = current_bid_for_position(pos, o)
                    break

        if current_price is None:
            continue

        entry = pos["entry_price"]
        city_name = LOCATIONS.get(mkt["city"], {}).get("name", mkt["city"])

        # Hours left to resolution
        end_date = mkt.get("event_end_date", "")
        hours_left = hours_to_resolution(end_date) if end_date else 999.0

        if pos.get("exit_policy") == EXIT_HOLD_TO_RESOLUTION:
            observed_high = get_metar(mkt["city"])
            if not near_lock_invalidated(pos, observed_high):
                continue
            decision_reason = "near_lock_invalidated"
        else:
            decision = evaluate_price_exit(
                entry_price=entry,
                current_price=current_price,
                hours_left=hours_left,
                exit_policy=pos.get("exit_policy", "standard"),
                stop_price=pos.get("stop_price"),
                trailing_activated=bool(pos.get("trailing_activated", False)),
                config=STRATEGY_CONFIG,
            )
            pos["stop_price"] = decision.stop_price
            pos["trailing_activated"] = decision.trailing_activated
            decision_reason = decision.reason

        if decision_reason is None:
            continue
        balance, did_close = close_position(
            pos,
            balance=balance,
            current_price=current_price,
            reason=decision_reason,
            closed_at=datetime.now(timezone.utc).isoformat(),
        )
        if not did_close:
            continue
        pnl = pos["pnl"]
        reason = {
            "take_profit": "TAKE",
            "stop_loss": "STOP",
            "trailing_stop": "TRAILING",
            "near_lock_invalidated": "INVALIDATED",
        }.get(decision_reason, "CLOSE")
        closed += 1
        print(f"  [{reason}] {city_name} {mkt['date']} | entry ${entry:.3f} exit ${current_price:.3f} | {hours_left:.0f}h left | PnL: {'+'if pnl>=0 else ''}{pnl:.2f}")
        save_market(mkt)

    if closed:
        state["balance"] = round(balance, 2)
        save_state(state)

    return closed


def run_loop():
    global _cal
    _cal = load_cal()
    original_stdout, original_stderr, log_file = install_paper_logging()

    try:
        print_section("WeatherBet operator console", "paper trading")
        print(f"  Cities       {len(LOCATIONS)}")
        print(f"  Balance      ${BALANCE:,.0f} | max bet ${MAX_BET}")
        print(f"  Cadence      scan {ACTIVE_SCAN_INTERVAL//60}m | monitor {MONITOR_INTERVAL//60}m")
        print(f"  Sources      ECMWF + HRRR(US) + METAR(D+0)")
        print(f"  Data         {DATA_DIR.resolve()}")
        print(f"  Log          {PAPER_LOG_FILE.resolve()}")
        print(f"  Stop         Ctrl+C")

        last_full_scan = 0

        while True:
            now_ts  = time.time()
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Full scan once per hour
            if now_ts - last_full_scan >= ACTIVE_SCAN_INTERVAL:
                print_section("Full scan", now_str)
                try:
                    new_pos, closed, resolved = scan_and_update()
                    state = load_state()
                    print(f"\n  {color('Summary', 'bold')}")
                    print(f"      Balance   ${state['balance']:,.2f}")
                    print(f"      Activity  {new_pos} new | {closed} closed | {resolved} resolved")
                    last_full_scan = time.time()
                except KeyboardInterrupt:
                    print(f"\n  {badge('WARN')} Stopping — saving state...")
                    save_state(load_state())
                    print(f"  {color('Done.', 'green')}")
                    break
                except requests.exceptions.ConnectionError:
                    print(f"  {badge('WARN')} Connection lost — waiting 60 sec")
                    time.sleep(60)
                    continue
                except Exception as e:
                    print(f"  {badge('ERROR')} {e} — waiting 60 sec")
                    time.sleep(60)
                    continue
            else:
                # Quick stop monitoring
                print(f"\n{color('Monitor', 'bold')}  {color(now_str, 'dim')}")
                try:
                    stopped = monitor_positions()
                    if stopped:
                        state = load_state()
                        print(f"  Balance ${state['balance']:,.2f}")
                    else:
                        print(f"  {color('No stop/take-profit exits.', 'dim')}")
                except Exception as e:
                    print(f"  {badge('ERROR')} Monitor error: {e}")

            try:
                time.sleep(MONITOR_INTERVAL)
            except KeyboardInterrupt:
                print(f"\n  Stopping — saving state...")
                save_state(load_state())
                print(f"  Done. Bye!")
                break
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()

# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "run":
        run_loop()
    elif cmd == "status":
        _cal = load_cal()
        print_status()
    elif cmd == "report":
        _cal = load_cal()
        print_report()
    elif cmd == "validate":
        _cal = load_cal()
        print_validation_report()
    elif cmd == "archive-reset":
        try:
            archive_and_reset_paper()
        except ValueError as error:
            print(f"  {badge('ERROR')} {error}")
            sys.exit(1)
    elif cmd == "repair-calibration":
        mode = sys.argv[2] if len(sys.argv) > 2 else "--dry-run"
        if mode not in {"--dry-run", "--apply"}:
            print("Usage: python weatherbet.py repair-calibration [--dry-run|--apply]")
            sys.exit(2)
        _cal = load_cal()
        repair_calibration(apply_changes=mode == "--apply")
    else:
        print("Usage: python weatherbet.py [run|status|report|validate|archive-reset|repair-calibration]")
