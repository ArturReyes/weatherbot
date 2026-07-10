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
from validation import PromotionPolicy, chronological_out_of_sample_report

from calibration import (
    LEAD_TIME_BUCKETS,
    bias_adjusted_forecast,
    calibration_errors,
    decaying_mean_error,
    lead_time_bucket,
    rmse_sigma,
)
from paper_trading import close_position, market_quote, revalidate_signal, yes_quote
from strategy import (
    EXIT_HOLD_TO_RESOLUTION,
    NO,
    YES,
    BucketQuote,
    ForecastContext,
    StrategyCandidate,
    StrategyConfig,
    generate_strategy_candidates,
    source_spread_from_values,
)
from trading_risk import (
    RiskLimits,
    assess_trade_risk,
    fee_adjusted_ev,
    fee_adjusted_kelly,
    market_fee_rate,
)

# =============================================================================
# CONFIG
# =============================================================================

load_dotenv()

with open("config.json", encoding="utf-8") as f:
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
SCAN_INTERVAL    = _cfg.get("scan_interval", 3600)   # every hour
OPPORTUNITY_SCAN_INTERVAL = int(_cfg.get("opportunity_scan_interval_seconds", 300))
ACTIVE_SCAN_INTERVAL = min(SCAN_INTERVAL, OPPORTUNITY_SCAN_INTERVAL)
CALIBRATION_MIN  = _cfg.get("calibration_min", 15)
BIAS_DECAY = float(_cfg.get("bias_decay", 0.97))
BIAS_PRIOR_STRENGTH = float(_cfg.get("bias_prior_strength", 20.0))
MAX_BIAS_CORRECTION_F = float(_cfg.get("max_bias_correction_f", 3.0))
MAX_BIAS_CORRECTION_C = float(_cfg.get("max_bias_correction_c", 1.5))
# VC_KEY: fetch from env var first, fall back to config.json
VC_KEY           = os.environ.get("VC_KEY") or _cfg.get("vc_key", "")

SIGMA_F = 2.0
SIGMA_C = 1.2

DATA_DIR         = Path("data")
DATA_DIR.mkdir(exist_ok=True)
STATE_FILE       = DATA_DIR / "state.json"
MARKETS_DIR      = DATA_DIR / "markets"
MARKETS_DIR.mkdir(exist_ok=True)
CALIBRATION_FILE = DATA_DIR / "calibration.json"
PAPER_LOG_FILE   = Path(_cfg.get("paper_log_file", "paper_trading.log"))
STRATEGY_CONFIG  = StrategyConfig.from_mapping(
    _cfg,
    min_ev=MIN_EV,
    max_price=MAX_PRICE,
    max_slippage=MAX_SLIPPAGE,
)
PAPER_RISK_LIMITS = RiskLimits(
    max_total_exposure_pct=float(_cfg.get("max_total_exposure_pct", 0.25)),
    max_event_exposure_pct=float(_cfg.get("max_event_exposure_pct", 0.10)),
    max_daily_loss_pct=float(_cfg.get("max_daily_loss_pct", 0.05)),
    max_open_positions=int(_cfg.get("max_open_positions", 5)),
    max_signal_age_seconds=float(_cfg.get("max_signal_age_seconds", 120)),
)
PROMOTION_POLICY = PromotionPolicy.from_mapping(_cfg)

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
    "paris":        {"lat": 48.9962,  "lon":    2.5979, "name": "Paris",         "station": "LFPG", "unit": "C", "region": "eu"},
    "munich":       {"lat": 48.3537,  "lon":   11.7750, "name": "Munich",        "station": "EDDM", "unit": "C", "region": "eu"},
    "ankara":       {"lat": 40.1281,  "lon":   32.9951, "name": "Ankara",        "station": "LTAC", "unit": "C", "region": "eu"},
    "seoul":        {"lat": 37.4691,  "lon":  126.4505, "name": "Seoul",         "station": "RKSI", "unit": "C", "region": "asia"},
    "tokyo":        {"lat": 35.7647,  "lon":  140.3864, "name": "Tokyo",         "station": "RJTT", "unit": "C", "region": "asia"},
    "shanghai":     {"lat": 31.1443,  "lon":  121.8083, "name": "Shanghai",      "station": "ZSPD", "unit": "C", "region": "asia"},
    "singapore":    {"lat":  1.3502,  "lon":  103.9940, "name": "Singapore",     "station": "WSSS", "unit": "C", "region": "asia"},
    "lucknow":      {"lat": 26.7606,  "lon":   80.8893, "name": "Lucknow",       "station": "VILK", "unit": "C", "region": "asia"},
    "tel-aviv":     {"lat": 32.0114,  "lon":   34.8867, "name": "Tel Aviv",      "station": "LLBG", "unit": "C", "region": "asia"},
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
    "lucknow": "Asia/Kolkata", "tel-aviv": "Asia/Jerusalem",
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
    if lead_bucket:
        entry = _cal.get(f"{city_slug}_{source}_{lead_bucket}", {})

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
        "sigma": get_sigma(city_slug, source, lead_bucket),
    }

def run_calibration(markets):
    """Recalculates aggregate sigma and lead-bucket bias/sigma calibration."""
    resolved = [m for m in markets if m.get("resolved") and m.get("actual_temp") is not None]
    cal = load_cal()
    updated = []
    updated_at = datetime.now(timezone.utc).isoformat()
    lead_buckets = [label for _, _, label in LEAD_TIME_BUCKETS] + ["72h_plus"]

    # ``hrrr`` was the old GFS-seamless label.  Real HRRR is stored under the
    # explicit source name below, so historical records cannot contaminate it.
    for source in ["ecmwf", "hrrr_conus", "metar"]:
        for city in set(m["city"] for m in resolved):
            errors = calibration_errors(resolved, city=city, source=source)
            if len(errors) < CALIBRATION_MIN:
                continue
            key  = f"{city}_{source}"
            old  = cal.get(key, {}).get("sigma", SIGMA_F if LOCATIONS[city]["unit"] == "F" else SIGMA_C)
            floor = 0.5 if LOCATIONS[city]["unit"] == "F" else 0.25
            new  = round(rmse_sigma(errors, floor=floor), 3)
            cal[key] = {"sigma": new, "n": len(errors), "updated_at": updated_at}
            if abs(new - old) > 0.05:
                updated.append(f"{LOCATIONS[city]['name']} {source}: {old:.2f}->{new:.2f}")

            for lead_bucket in lead_buckets:
                bucket_errors = calibration_errors(
                    resolved,
                    city=city,
                    source=source,
                    lead_bucket=lead_bucket,
                )
                if len(bucket_errors) < CALIBRATION_MIN:
                    continue
                estimate = decaying_mean_error(
                    resolved,
                    city=city,
                    source=source,
                    lead_bucket=lead_bucket,
                    decay=BIAS_DECAY,
                    prior_strength=BIAS_PRIOR_STRENGTH,
                )
                bucket_key = f"{city}_{source}_{lead_bucket}"
                cal[bucket_key] = {
                    "city": city,
                    "source": source,
                    "lead_bucket": lead_bucket,
                    "bias": round(estimate.bias, 3),
                    "raw_bias": round(estimate.raw_bias, 3),
                    "sigma": round(rmse_sigma(bucket_errors, floor=floor), 3),
                    "n": len(bucket_errors),
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
    for attempt in range(3):
        try:
            data = requests.get(url, timeout=(5, 10)).json()
            if "error" not in data:
                for date, temp in zip(data["daily"]["time"], data["daily"]["temperature_2m_max"]):
                    if date in dates and temp is not None:
                        result[date] = round(temp, 1) if unit == "C" else round(temp)
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
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
        f"&models=hrrr_conus"
    )
    for attempt in range(3):
        try:
            data = requests.get(url, timeout=(5, 10)).json()
            if "error" not in data:
                for date, temp in zip(data["daily"]["time"], data["daily"]["temperature_2m_max"]):
                    if date in dates and temp is not None:
                        result[date] = round(temp)
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(3)
            else:
                print(f"  [HRRR] {city_slug}: {e}")
    return result

def get_metar(city_slug):
    """Current observed temperature from METAR station. D+0 only."""
    loc = LOCATIONS[city_slug]
    station = loc["station"]
    unit = loc["unit"]
    try:
        url = f"https://aviationweather.gov/api/data/metar?ids={station}&format=json"
        data = requests.get(url, timeout=(5, 8)).json()
        if data and isinstance(data, list):
            temp_c = data[0].get("temp")
            if temp_c is not None:
                if unit == "F":
                    return round(float(temp_c) * 9/5 + 32)
                return round(float(temp_c), 1)
    except Exception as e:
        print(f"  [METAR] {city_slug}: {e}")
    return None

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
        data = requests.get(url, timeout=(5, 10)).json()
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
    except Exception as e:
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

def check_market_resolved(market_id):
    """
    Checks if the market closed on Polymarket and who won.
    Returns: None (still open), True (YES won), False (NO won)
    """
    try:
        r = requests.get(f"https://gamma-api.polymarket.com/markets/{market_id}", timeout=(5, 8))
        data = r.json()
        closed = data.get("closed", False)
        if not closed:
            return None
        # Check YES price — if ~1.0 then WIN, if ~0.0 then LOSS
        prices = json.loads(data.get("outcomePrices", "[0.5,0.5]"))
        yes_price = float(prices[0])
        if yes_price >= 0.95:
            return True   # WIN
        elif yes_price <= 0.05:
            return False  # LOSS
        return None  # not yet determined
    except Exception as e:
        print(f"  [RESOLVE] {market_id}: {e}")
    return None

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
        "hours_at_discovery": round(hours, 1),
        "status":             "open",           # open | closed | resolved
        "position":           None,             # filled when position opens
        "actual_temp":        None,             # filled after resolution
        "resolved_outcome":   None,             # win / loss / no_position
        "pnl":                None,
        "forecast_snapshots": [],               # list of forecast snapshots
        "market_snapshots":   [],               # list of market price snapshots
        "all_outcomes":       [],               # all market buckets
        "created_at":         datetime.now(timezone.utc).isoformat(),
    }

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
    if size < 0.50:
        return None
    total_cost_per_share = candidate.entry_price + candidate.fee_rate * candidate.entry_price * (1.0 - candidate.entry_price)
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
        "strategy_reason": candidate.reason,
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

            # Skip if market already resolved
            if mkt["status"] == "resolved":
                continue

            # Update outcomes list — prices taken directly from event
            outcomes = []
            for market in event.get("markets", []):
                question = market.get("question", "")
                mid      = str(market.get("id", ""))
                volume   = float(market.get("volume", 0))
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
                    stop  = pos.get("stop_price", entry * 0.80)  # 20% stop by default

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
                    elif pos.get("exit_policy") != EXIT_HOLD_TO_RESOLUTION:
                        # Trailing: if up 20%+ — move stop to breakeven
                        if current_price >= entry * 1.20 and stop < entry:
                            pos["stop_price"] = entry
                            pos["trailing_activated"] = True

                        # Check stop
                        if current_price <= stop:
                            balance, did_close = close_position(
                                pos,
                                balance=balance,
                                current_price=current_price,
                                reason="stop_loss" if current_price < entry else "trailing_stop",
                                closed_at=snap.get("ts"),
                            )
                            if did_close:
                                closed += 1
                                reason = "STOP" if current_price < entry else "TRAILING BE"
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
            if not mkt.get("position") and forecast_temp is not None and hours >= MIN_HOURS:
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
                )
                candidates = generate_strategy_candidates(
                    buckets=buckets,
                    context=context,
                    config=STRATEGY_CONFIG,
                )
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

                if best_signal and best_candidate is not None:
                    # Fetch real bestAsk from Polymarket API for accurate entry price
                    skip_position = False
                    try:
                        r = requests.get(f"https://gamma-api.polymarket.com/markets/{best_signal['market_id']}", timeout=(3, 5))
                        mdata = r.json()
                        refreshed = revalidate_signal(
                            best_signal,
                            mdata,
                            min_ev=min_ev_for_candidate(best_candidate),
                            max_price=max_price_for_candidate(best_candidate),
                            max_spread=MAX_SLIPPAGE,
                        )
                        if refreshed is None:
                            quote = market_quote(mdata, best_signal.get("outcome_side", YES))
                            city_events.append(scan_event(
                                "SKIP",
                                f"{date} | refreshed ask ${quote.ask:.3f}, spread ${quote.ask - quote.bid:.3f}, or EV below minimum",
                            ))
                            skip_position = True
                        else:
                            best_signal = refreshed
                    except Exception as e:
                        city_events.append(scan_event(
                            "WARN",
                            f"could not fetch real ask for {best_signal['market_id']}: {e}",
                        ))
                        skip_position = True

                    if not skip_position and best_signal["entry_price"] < max_price_for_candidate(best_candidate):
                        paper_positions = [
                            market["position"]
                            for market in load_all_markets()
                            if market.get("position") is not None
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
        if mkt["status"] == "resolved":
            continue

        pos = mkt.get("position")
        if not pos or pos.get("status") != "open":
            continue

        market_id = pos.get("market_id")
        if not market_id:
            continue

        # Check if market closed on Polymarket
        yes_won = check_market_resolved(market_id)
        if yes_won is None:
            continue  # market still open
        side = str(pos.get("outcome_side") or pos.get("side") or YES).upper()
        won = bool(yes_won) if side == YES else not bool(yes_won)

        # Market closed — record result
        price  = pos["entry_price"]
        size   = pos["cost"]
        shares = pos["shares"]
        pnl    = round(shares * (1 - price), 2) if won else round(-size, 2)

        balance += size + pnl
        pos["exit_price"]   = 1.0 if won else 0.0
        pos["pnl"]          = pnl
        pos["close_reason"] = "resolved"
        pos["closed_at"]    = now.isoformat()
        pos["status"]       = "closed"
        mkt["pnl"]          = pnl
        mkt["status"]       = "resolved"
        mkt["resolved_outcome"] = "win" if won else "loss"

        if won:
            state["wins"] += 1
        else:
            state["losses"] += 1

        result = "WIN" if won else "LOSS"
        resolution_events.append(scan_event(
            result,
            f"{mkt['city_name']} {mkt['date']} | PnL {'+' if pnl >= 0 else ''}{pnl:.2f}",
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
    resolved_count = len([m for m in all_mkts if m["status"] == "resolved"])
    if resolved_count >= CALIBRATION_MIN:
        global _cal
        _cal = run_calibration(all_mkts)

    return new_pos, closed, resolved

# =============================================================================
# REPORT
# =============================================================================

def print_status():
    state    = load_state()
    markets  = load_all_markets()
    open_pos = [m for m in markets if m.get("position") and m["position"].get("status") == "open"]
    resolved = [m for m in markets if m["status"] == "resolved" and m.get("pnl") is not None]

    bal     = state["balance"]
    start   = state["starting_balance"]
    ret_pct = (bal - start) / start * 100
    wins    = state["wins"]
    losses  = state["losses"]
    total   = wins + losses

    print(f"\n{'='*55}")
    print(f"  WEATHERBET — STATUS")
    print(f"{'='*55}")
    print(f"  Balance:     ${bal:,.2f}  (start ${start:,.2f}, {'+'if ret_pct>=0 else ''}{ret_pct:.1f}%)")
    print(f"  Trades:      {total} | W: {wins} | L: {losses} | WR: {wins/total:.0%}" if total else "  No trades yet")
    print(f"  Open:        {len(open_pos)}")
    print(f"  Resolved:    {len(resolved)}")

    if open_pos:
        print(f"\n  Open positions:")
        total_unrealized = 0.0
        for m in open_pos:
            pos      = m["position"]
            unit_sym = "F" if m["unit"] == "F" else "C"
            label    = f"{pos['bucket_low']}-{pos['bucket_high']}{unit_sym}"

            # Current price from latest market snapshot
            current_price = pos["entry_price"]
            snaps = m.get("market_snapshots", [])
            if snaps:
                # Find our bucket price in all_outcomes
                for o in m.get("all_outcomes", []):
                    if o["market_id"] == pos["market_id"]:
                        current_price = o["price"]
                        break

            unrealized = round((current_price - pos["entry_price"]) * pos["shares"], 2)
            total_unrealized += unrealized
            pnl_str = f"{'+'if unrealized>=0 else ''}{unrealized:.2f}"

            print(f"    {m['city_name']:<16} {m['date']} | {label:<14} | "
                  f"entry ${pos['entry_price']:.3f} -> ${current_price:.3f} | "
                  f"PnL: {pnl_str} | {pos['forecast_src'].upper()}")

        sign = "+" if total_unrealized >= 0 else ""
        print(f"\n  Unrealized PnL: {sign}{total_unrealized:.2f}")

    print(f"{'='*55}\n")

def print_report():
    markets  = load_all_markets()
    resolved = [m for m in markets if m["status"] == "resolved" and m.get("pnl") is not None]

    print(f"\n{'='*55}")
    print(f"  WEATHERBET — FULL REPORT")
    print(f"{'='*55}")

    if not resolved:
        print("  No resolved markets yet.")
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
    print("  Note: paper fills are assumed; post-entry price movement is not reported as CLV.")
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

        # Fetch real bestBid from Polymarket API — actual sell price
        current_price = None
        try:
            r = requests.get(f"https://gamma-api.polymarket.com/markets/{mid}", timeout=(3, 5))
            mdata = r.json()
            best_bid = mdata.get("bestBid")
            if best_bid is not None:
                current_price = float(best_bid)
        except Exception:
            pass

        # Fallback to cached price if API failed
        if current_price is None:
            for o in mkt.get("all_outcomes", []):
                if o["market_id"] == mid:
                    current_price = o.get("bid", o["price"])
                    break

        if current_price is None:
            continue

        entry = pos["entry_price"]
        stop  = pos.get("stop_price", entry * 0.80)
        city_name = LOCATIONS.get(mkt["city"], {}).get("name", mkt["city"])

        # Hours left to resolution
        end_date = mkt.get("event_end_date", "")
        hours_left = hours_to_resolution(end_date) if end_date else 999.0

        # Take-profit threshold based on hours to resolution
        if hours_left < 24:
            take_profit = None        # hold to resolution
        elif hours_left < 48:
            take_profit = 0.85        # 24-48h: take profit at $0.85
        else:
            take_profit = 0.75        # 48h+: take profit at $0.75

        # Trailing: if up 20%+ — move stop to breakeven
        if current_price >= entry * 1.20 and stop < entry:
            pos["stop_price"] = entry
            pos["trailing_activated"] = True
            print(f"  [TRAILING] {city_name} {mkt['date']} — stop moved to breakeven ${entry:.3f}")

        # Check take-profit
        take_triggered = take_profit is not None and current_price >= take_profit
        # Check stop
        stop_triggered = current_price <= stop

        if take_triggered or stop_triggered:
            pnl = round((current_price - entry) * pos["shares"], 2)
            balance += pos["cost"] + pnl
            pos["closed_at"]    = datetime.now(timezone.utc).isoformat()
            if take_triggered:
                pos["close_reason"] = "take_profit"
                reason = "TAKE"
            elif current_price < entry:
                pos["close_reason"] = "stop_loss"
                reason = "STOP"
            else:
                pos["close_reason"] = "trailing_stop"
                reason = "TRAILING BE"
            pos["exit_price"]   = current_price
            pos["pnl"]          = pnl
            pos["status"]       = "closed"
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
    else:
        print("Usage: python weatherbet.py [run|status|report|validate]")
