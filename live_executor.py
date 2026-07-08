#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
live_executor.py — Live Trading Execution Layer for Weatherbet
===============================================================

Connects the weather strategy engine (weatherbet.py) to Polymarket's CLOB
via the new unified Python SDK (polymarket-client).

Architecture
------------
  weatherbet.py  ──>  (math, forecasts, calibration)  ──>  live_executor.py
                        (strategy signals)                      │
                                                                 ├─ SecureClient → CLOB orders
                                                                 ├─ PublicClient → on-chain state
                                                                 └─ Telegram notifications

Requirements
------------
  pip install --pre polymarket-client
  pip install python-dotenv requests

Environment Variables (.env)
-----------------------------
  PK                     Polygon EOA private key (0x...)
  WALLET                 Wallet address (optional; derived from PK if omitted)
  TELEGRAM_BOT_TOKEN     Telegram integration (optional)
  TELEGRAM_CHAT_ID       Telegram integration (optional)
  VC_KEY                 Visual Crossing API key (optional)

Strategy Parameters (config.json)
----------------------------------
  balance           Starting balance for Kelly sizing (default: 10000)
  max_bet           Max bet per trade in USDC (default: 20)
  min_ev            Minimum edge required (default: 0.10)
  max_price         Max entry price (default: 0.45)
  max_slippage      Max spread fraction (default: 0.03)
  kelly_fraction    Fraction of Kelly to bet (default: 0.25)
  scan_interval     Seconds between full scans (default: 1800)
  calibration_min   Minimum resolved markets per city (default: 15)

Usage
-----
  python live_executor.py          # run the live trading loop
  python live_executor.py status   # show live positions and balance
  python live_executor.py cancel   # cancel all open orders
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

# ── Polymarket SDK ────────────────────────────────────────────────
# Ensure available before importing weatherbet (which doesn't need it)
try:
    from polymarket import SecureClient, PublicClient
    from polymarket.environments import PRODUCTION
    from polymarket.models import OrderSide
except ImportError:
    SecureClient = None  # type: ignore
    print(
        "Error: polymarket-client SDK not installed.\n"
        "  pip install --pre polymarket-client",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Shared Strategy Engine ────────────────────────────────────────
# Import the fixed math, forecasts, and calibration from weatherbet.py
# This keeps the strategy logic in ONE place (weatherbet.py) and the
# execution layer separate. Both can be updated independently.
sys.path.insert(0, str(Path(__file__).parent.resolve()))
import weatherbet  # noqa: E402
from live_trading import (  # noqa: E402
    PolymarketGateway,
    ProcessLock,
    ProcessLockError,
    ReconciliationReport,
    reconcile_entry,
    reconcile_exit,
    reconcile_state,
)
from trading_risk import (  # noqa: E402
    RiskLimits,
    assess_trade_risk,
    contract_matches_strategy,
    extract_market_date,
    market_fee_rate,
)

# Re-export key strategy bindings for readability
LOCATIONS = weatherbet.LOCATIONS
SIGMA_F = weatherbet.SIGMA_F
SIGMA_C = weatherbet.SIGMA_C
norm_cdf = weatherbet.norm_cdf
bucket_prob = weatherbet.bucket_prob
calc_ev = weatherbet.calc_ev
calc_kelly = weatherbet.calc_kelly
bet_size = weatherbet.bet_size
get_sigma = weatherbet.get_sigma
load_cal = weatherbet.load_cal
get_ecmwf = weatherbet.get_ecmwf
get_hrrr = weatherbet.get_hrrr
get_metar = weatherbet.get_metar
forecast_calibration = weatherbet.forecast_calibration

# =============================================================================
# CONFIG
# =============================================================================

load_dotenv()

PRIVATE_KEY = os.environ.get("PK", "")
WALLET_ADDR = os.environ.get("WALLET", None)
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

CONFIG_FILE = Path("config.json")
if CONFIG_FILE.exists():
    with open(CONFIG_FILE, encoding="utf-8") as f:
        _cfg = json.load(f)
else:
    _cfg = {}

BALANCE_REF = Decimal(str(_cfg.get("balance", 10000.0)))
MAX_BET = float(_cfg.get("max_bet", 20.0))
MIN_EV = float(_cfg.get("min_ev", 0.10))
MAX_PRICE = float(_cfg.get("max_price", 0.45))
MIN_VOLUME = float(_cfg.get("min_volume", 500))
MIN_HOURS = float(_cfg.get("min_hours", 2.0))
MAX_HOURS = float(_cfg.get("max_hours", 72.0))
KELLY_FRACTION = float(_cfg.get("kelly_fraction", 0.25))
MAX_SLIPPAGE = float(_cfg.get("max_slippage", 0.03))
SCAN_INTERVAL = int(_cfg.get("scan_interval", 1800))
FORECAST_CACHE_TTLS = {
    "ecmwf": float(_cfg.get("forecast_cache_ttl_ecmwf_seconds", 1800)),
    "hrrr": float(_cfg.get("forecast_cache_ttl_hrrr_seconds", 600)),
    "metar": float(_cfg.get("forecast_cache_ttl_metar_seconds", 45)),
}
RISK_LIMITS = RiskLimits(
    max_total_exposure_pct=float(_cfg.get("max_total_exposure_pct", 0.25)),
    max_event_exposure_pct=float(_cfg.get("max_event_exposure_pct", 0.10)),
    max_daily_loss_pct=float(_cfg.get("max_daily_loss_pct", 0.05)),
    max_open_positions=int(_cfg.get("max_open_positions", 5)),
    max_signal_age_seconds=float(_cfg.get("max_signal_age_seconds", 120)),
)

DATA_DIR = Path("data")
LIVE_STATE_FILE = DATA_DIR / "live_state.json"
LIVE_LOCK_FILE = DATA_DIR / "live_executor.lock"

# =============================================================================
# LOGGING
# =============================================================================

logger = logging.getLogger("live_executor")
logger.setLevel(logging.DEBUG)

_fh = logging.FileHandler("live_trading.log", encoding="utf-8", mode="a")
_fh.setLevel(logging.DEBUG)
_fh.setFormatter(
    logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
)

_sh = logging.StreamHandler(sys.stdout)
_sh.setLevel(logging.INFO)
_sh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logger.addHandler(_fh)
logger.addHandler(_sh)

# =============================================================================
# STATE MANAGEMENT (atomic writes)
# =============================================================================


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON atomically via tempfile + os.replace."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_live_state() -> dict:
    """Load live execution state (open orders, positions)."""
    if LIVE_STATE_FILE.exists():
        return json.loads(LIVE_STATE_FILE.read_text(encoding="utf-8"))
    return {
        "balance_ref": float(BALANCE_REF),
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "positions": [],
        "failed_signals": [],
    }


def save_live_state(state: dict) -> None:
    """Save live execution state atomically."""
    _atomic_write_json(LIVE_STATE_FILE, state)


# =============================================================================
# TELEGRAM
# =============================================================================


def send_telegram(msg: str) -> None:
    """Send a Telegram notification."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg},
            timeout=10,
        )
    except Exception as e:
        logger.warning("Telegram send failed: %s", e)


# =============================================================================
# MARKET DATA (Gamma API — shared with weatherbet.py style)
# =============================================================================

GAMMA_BASE = "https://gamma-api.polymarket.com"
_SESSION = requests.Session()


def fetch_outdoor_markets() -> list[dict]:
    """Fetch all open weather/outdoor markets from Gamma API."""
    slug_pages = [
        ("weather", 20),
        ("outdoor", 5),
    ]
    seen: set[str] = set()
    markets: list[dict] = []

    for slug, pages in slug_pages:
        for page in range(1, pages + 1):
            try:
                resp = _SESSION.get(
                    f"{GAMMA_BASE}/markets",
                    params={
                        "tag_slug": slug,
                        "limit": 100,
                        "page": page,
                        "closed": "false",
                        "archived": "false",
                    },
                    timeout=(5, 15),
                )
                resp.raise_for_status()
                batch = resp.json()
                if not batch:
                    break
                for m in batch:
                    mid = m.get("id", "")
                    if mid and mid not in seen:
                        seen.add(mid)
                        markets.append(m)
            except requests.RequestException as e:
                logger.warning("Gamma fetch %s page %d: %s", slug, page, e)
                break
            time.sleep(0.05)  # rate-limit courtesy

    logger.info("Fetched %d open markets from Gamma", len(markets))
    return markets


def fetch_market_detail(market_id: str) -> dict | None:
    """Fetch a single market's detail including outcomePrices."""
    try:
        resp = _SESSION.get(
            f"{GAMMA_BASE}/markets/{market_id}", timeout=(5, 10)
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.warning("Failed to fetch market %s: %s", market_id, e)
        return None


# =============================================================================
# FORECAST CACHE  (weather data only; never market prices/orderbooks)
# =============================================================================


@dataclass
class CachedForecast:
    value: float | None
    fetched_at: float
    ttl_seconds: float

    def fresh_at(self, now_ts: float) -> bool:
        return now_ts - self.fetched_at < self.ttl_seconds


class ForecastCache:
    """Short-lived, source-aware cache for forecast inputs.

    This intentionally caches only weather inputs. Market prices and executable
    CLOB quotes stay outside this component and are refreshed before orders.
    """

    def __init__(
        self,
        *,
        fetch_ecmwf: Callable[[str, set[str]], dict[str, float]],
        fetch_hrrr: Callable[[str, set[str]], dict[str, float]],
        fetch_metar: Callable[[str], float | None],
        ttl_seconds: dict[str, float],
        now_fn: Callable[[], float] = time.time,
    ) -> None:
        self._fetch_ecmwf = fetch_ecmwf
        self._fetch_hrrr = fetch_hrrr
        self._fetch_metar = fetch_metar
        self._ttl_seconds = ttl_seconds
        self._now_fn = now_fn
        self._cache: dict[tuple[str, str, str], CachedForecast] = {}

    def sources_for(
        self,
        *,
        city_slug: str,
        date_str: str,
        is_us_city: bool,
        hours_to_resolution_value: float,
    ) -> list[tuple[str, float]]:
        sources: list[tuple[str, float]] = []

        ecmwf = self._daily("ecmwf", city_slug, date_str, self._fetch_ecmwf)
        if ecmwf is not None:
            sources.append(("ecmwf", ecmwf))

        if is_us_city:
            hrrr = self._daily("hrrr", city_slug, date_str, self._fetch_hrrr)
            if hrrr is not None:
                sources.append(("hrrr", hrrr))

        if hours_to_resolution_value < 24:
            metar = self._metar(city_slug)
            if metar is not None:
                sources.append(("metar", metar))

        return sources

    def _daily(
        self,
        source: str,
        city_slug: str,
        date_str: str,
        fetch: Callable[[str, set[str]], dict[str, float]],
    ) -> float | None:
        key = (source, city_slug, date_str)
        now_ts = self._now_fn()
        cached = self._cache.get(key)
        if cached is not None and cached.fresh_at(now_ts):
            return cached.value

        data = fetch(city_slug, {date_str})
        value = data.get(date_str)
        numeric_value = float(value) if value is not None else None
        self._cache[key] = CachedForecast(
            numeric_value,
            fetched_at=now_ts,
            ttl_seconds=self._ttl_seconds[source],
        )
        return numeric_value

    def _metar(self, city_slug: str) -> float | None:
        key = ("metar", city_slug, "")
        now_ts = self._now_fn()
        cached = self._cache.get(key)
        if cached is not None and cached.fresh_at(now_ts):
            return cached.value

        value = self._fetch_metar(city_slug)
        numeric_value = float(value) if value is not None else None
        self._cache[key] = CachedForecast(
            numeric_value,
            fetched_at=now_ts,
            ttl_seconds=self._ttl_seconds["metar"],
        )
        return numeric_value


def build_forecast_cache() -> ForecastCache:
    return ForecastCache(
        fetch_ecmwf=get_ecmwf,
        fetch_hrrr=get_hrrr,
        fetch_metar=get_metar,
        ttl_seconds=FORECAST_CACHE_TTLS,
    )


# =============================================================================
# SIGNAL GENERATION  (reuses weatherbet.py's strategy math)
# =============================================================================


@dataclass
class TradeSignal:
    """A verified strategy signal ready for execution."""

    action: str  # "BUY" or "SELL"
    token_id: str
    market_id: str
    condition_id: str
    city_slug: str
    city_name: str
    date_str: str
    forecast_temp: float
    bucket_low: float
    bucket_high: float
    unit: str
    probability: float
    entry_price: float
    spread: float
    ev: float
    kelly: float
    size_usdc: float
    shares: float
    forecast_source: str
    sigma: float
    fee_rate: float = 0.0
    raw_forecast_temp: float | None = None
    corrected_forecast_temp: float | None = None
    forecast_bias: float = 0.0
    forecast_raw_bias: float = 0.0
    forecast_lead_bucket: str | None = None
    forecast_calibration_n: int = 0
    raw_probability: float | None = None
    raw_ev: float | None = None
    reason: str = ""
    created_at_ts: float = field(default_factory=time.time)


def parse_gamma_outcomes(market: dict) -> list[dict]:
    """Parse outcomes from a Gamma market dict into normalised form.

    Returns list of dicts with keys:
        market_id, token_id, price, bid, ask, volume, outcome
    """
    outcomes_raw = market.get("outcomes", json.dumps([]))
    if isinstance(outcomes_raw, str):
        try:
            outcomes_raw = json.loads(outcomes_raw)
        except (json.JSONDecodeError, TypeError):
            return []

    prices_raw = market.get("outcomePrices", "[0.5,0.5]")
    if isinstance(prices_raw, str):
        try:
            prices = json.loads(prices_raw)
        except (json.JSONDecodeError, TypeError):
            prices = [0.5, 0.5]
    else:
        prices = prices_raw

    clob_token_ids = market.get("clobTokenIds", "[]")
    if isinstance(clob_token_ids, str):
        try:
            clob_token_ids = json.loads(clob_token_ids)
        except (json.JSONDecodeError, TypeError):
            clob_token_ids = []

    result = []
    for i, outcome in enumerate(outcomes_raw):
        token_id = (
            clob_token_ids[i]
            if i < len(clob_token_ids) and clob_token_ids[i]
            else ""
        )
        price = float(prices[i]) if i < len(prices) else 0.5
        result.append(
            {
                "market_id": market.get("id"),
                "token_id": token_id,
                "price": price,
                "bid": float(market.get("bestBid", price)),
                "ask": float(market.get("bestAsk", price)),
                "volume": float(market.get("volume", 0)),
                "outcome": outcome,
            }
        )
    return result


def hours_to_resolution(end_date_iso: str) -> float:
    """Hours from now until the event end date."""
    try:
        end = datetime.fromisoformat(end_date_iso.replace("Z", "+00:00"))
        return max(0.0, (end - datetime.now(timezone.utc)).total_seconds() / 3600)
    except (ValueError, AttributeError):
        return 999.0


def generate_signals(
    markets: list[dict],
    state: dict,
    *,
    forecast_cache: ForecastCache | None = None,
) -> list[TradeSignal]:
    """Run the weather strategy and return executable trade signals.

    This function mirrors weatherbet.py's scan_and_update logic but returns
    structured TradeSignal objects instead of simulating trades in state.
    """
    signals: list[TradeSignal] = []
    cache = forecast_cache or build_forecast_cache()
    existing_tokens = {
        pos["token_id"]
        for pos in state.get("positions", [])
        if pos.get("status")
        in (
            "submitting",
            "pending",
            "unknown",
            "open",
            "exit_pending",
            "exit_unknown",
            "redeeming",
            "redemption_unknown",
            "redemption_confirmed",
            "missing",
            "unmanaged",
        )
    }
    for market in markets:
        try:
            # ── Parse market metadata ────────────────────────────
            question = market.get("question", "")
            slug = market.get("slug", "")
            mid = market.get("id", "")
            end_date = market.get("endDate", "") or market.get("eventEndDate", "")

            # Identify city + date from question slug
            # Weather markets look like: "will-it-be-X-degrees-in-CITY-on-DATE"
            parts = slug.split("-")
            # heuristics: city is typically the 6th-to-last word group
            # fallback: scan question for known city names
            city_slug = None
            for key, loc in LOCATIONS.items():
                if loc["name"].lower().replace(" ", "-") in slug.lower() or loc["name"].lower() in question.lower():
                    city_slug = key
                    break

            if city_slug is None:
                continue  # not a weather market we track

            # Extract date from slug (last 3 groups: YYYY-MM-DD)
            date_str = _extract_date_from_slug(slug) or _extract_date_from_question(question)
            if date_str is None:
                continue

            loc = LOCATIONS[city_slug]
            unit = loc["unit"]
            unit_sym = "F" if unit == "F" else "C"

            contract = contract_matches_strategy(
                market,
                city_name=loc["name"],
                station=loc["station"],
                unit=unit_sym,
                date_str=date_str,
            )
            if not contract.valid:
                logger.debug("Rejected contract %s: %s", mid, contract.reason)
                continue

            # ── Check time horizon ───────────────────────────────
            hrs = hours_to_resolution(end_date)
            if hrs < MIN_HOURS or hrs > MAX_HOURS:
                continue

            # ── Check volume filter ──────────────────────────────
            if float(market.get("volume", 0)) < MIN_VOLUME:
                continue

            # ── Parse temperature bucket from question ───────────
            bucket = weatherbet.parse_temp_range(question)
            if bucket is None:
                continue
            t_low, t_high = bucket

            # ── Parse outcomes & find our token ──────────────────
            outcomes = parse_gamma_outcomes(market)
            our_outcome = None
            non_outcome = None
            for o in outcomes:
                out_name = o.get("outcome", "").lower()
                if "yes" in out_name or "will" in out_name:
                    our_outcome = o
                else:
                    non_outcome = o
            if our_outcome is None and len(outcomes) >= 2:
                our_outcome = outcomes[0]  # fallback: first outcome = YES
                non_outcome = outcomes[1]

            if our_outcome is None or not our_outcome.get("token_id"):
                continue

            token_id = our_outcome["token_id"]
            if token_id in existing_tokens:
                continue  # already have a position

            condition_id = market.get("conditionId", market.get("condition_id", ""))
            ask = float(our_outcome.get("ask", our_outcome["price"]))
            bid = float(our_outcome.get("bid", our_outcome["price"]))
            spread = ask - bid
            fee_rate = market_fee_rate(market)

            # ── Check price + spread filters ─────────────────────
            if ask >= MAX_PRICE or spread > MAX_SLIPPAGE:
                continue

            # ── Get forecasts ────────────────────────────────────
            sources = cache.sources_for(
                city_slug=city_slug,
                date_str=date_str,
                is_us_city=loc["region"] == "us",
                hours_to_resolution_value=hrs,
            )

            if not sources:
                continue  # no forecast available

            # Prefer HRRR for near-term US, then ECMWF, then METAR
            priority = {"hrrr": 0, "ecmwf": 1, "metar": 2}
            sources.sort(key=lambda x: priority.get(x[0], 99))
            best_source, raw_forecast_temp = sources[0]
            calibration = forecast_calibration(
                city_slug,
                best_source,
                raw_forecast_temp,
                datetime.now(timezone.utc).isoformat(),
                end_date,
            )
            forecast_temp = calibration["corrected_forecast_temp"]

            # ── Compute probability ──────────────────────────────
            sigma = float(calibration["sigma"])
            raw_prob = bucket_prob(raw_forecast_temp, t_low, t_high, sigma)
            prob = bucket_prob(forecast_temp, t_low, t_high, sigma)

            # ── Compute EV ───────────────────────────────────────
            raw_ev = calc_ev(raw_prob, ask, fee_rate)
            ev = calc_ev(prob, ask, fee_rate)
            if ev < MIN_EV:
                continue

            # ── Kelly bet sizing ─────────────────────────────────
            kelly_frac = calc_kelly(prob, ask, fee_rate)
            balance = Decimal(str(state.get("balance_ref", 10000.0)))
            raw_size = bet_size(kelly_frac, float(balance))
            size_usdc = min(raw_size, MAX_BET)
            shares = round(size_usdc / ask, 2)

            if shares <= 0 or size_usdc <= 0:
                continue

            # ── Re-check with real ask price ─────────────────────
            try:
                detail = fetch_market_detail(mid)
                if detail:
                    detail_contract = contract_matches_strategy(
                        detail,
                        city_name=loc["name"],
                        station=loc["station"],
                        unit=unit_sym,
                        date_str=date_str,
                    )
                    if not detail_contract.valid:
                        continue
                    real_ask = float(detail.get("bestAsk", ask))
                    real_bid = float(detail.get("bestBid", bid))
                    fee_rate = market_fee_rate(detail)
                    real_spread = real_ask - real_bid
                    if real_spread > MAX_SLIPPAGE or real_ask >= MAX_PRICE:
                        continue
                    # Recalculate EV with real price
                    raw_ev = calc_ev(raw_prob, real_ask, fee_rate)
                    ev = calc_ev(prob, real_ask, fee_rate)
                    if ev < MIN_EV:
                        continue
                    ask = real_ask
                    bid = real_bid
                    spread = real_spread
                    kelly_frac = calc_kelly(prob, real_ask, fee_rate)
                    raw_size = bet_size(kelly_frac, float(balance))
                    size_usdc = min(raw_size, MAX_BET)
                    shares = round(size_usdc / real_ask, 2)
                    if shares <= 0 or size_usdc <= 0:
                        continue
            except Exception:
                pass

            # ── Build signal ─────────────────────────────────────
            signal = TradeSignal(
                action="BUY",
                token_id=token_id,
                market_id=mid,
                condition_id=condition_id,
                city_slug=city_slug,
                city_name=loc["name"],
                date_str=date_str,
                forecast_temp=forecast_temp,
                bucket_low=t_low,
                bucket_high=t_high,
                unit=unit_sym,
                probability=prob,
                entry_price=ask,
                spread=spread,
                ev=ev,
                kelly=kelly_frac,
                size_usdc=size_usdc,
                shares=shares,
                forecast_source=best_source,
                sigma=sigma,
                fee_rate=fee_rate,
                raw_forecast_temp=float(raw_forecast_temp),
                corrected_forecast_temp=float(forecast_temp),
                forecast_bias=float(calibration["forecast_bias"]),
                forecast_raw_bias=float(calibration["forecast_raw_bias"]),
                forecast_lead_bucket=calibration["forecast_lead_bucket"],
                forecast_calibration_n=int(calibration["forecast_calibration_n"]),
                raw_probability=raw_prob,
                raw_ev=raw_ev,
                reason=f"EV {ev:+.2f} @ ${ask:.3f}",
            )
            signals.append(signal)

        except Exception as e:
            logger.debug("Signal generation error for %s: %s", market.get("id"), e)
            continue

    # Sort by EV descending (best opportunities first)
    signals.sort(key=lambda s: s.ev, reverse=True)
    logger.info("Generated %d trade signals", len(signals))
    return signals


def _extract_date_from_slug(slug: str) -> str | None:
    """Extract YYYY-MM-DD date from the last 3 groups of the slug."""
    return extract_market_date(slug)


def _extract_date_from_question(question: str) -> str | None:
    """Try to extract a date string from the question text."""
    import re

    # Match "MMM DD YYYY" or "DD MMM YYYY"
    for pat in (
        r"(\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4})",
        r"((?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})",
    ):
        m = re.search(pat, question, re.IGNORECASE)
        if m:
            try:
                from dateutil import parser as dtparser
                return dtparser.parse(m.group(1)).strftime("%Y-%m-%d")
            except (ImportError, ValueError):
                return m.group(1)
    return None


def _token_quote_from_market(market: dict, token_id: str) -> tuple[float, float] | None:
    for outcome in parse_gamma_outcomes(market):
        if outcome.get("token_id") == token_id:
            return float(outcome["bid"]), float(outcome["ask"])
    if "bestBid" in market and "bestAsk" in market:
        return float(market["bestBid"]), float(market["bestAsk"])
    return None


def refresh_signal_with_live_market(
    signal: TradeSignal,
    market: dict,
    *,
    balance_ref: float,
) -> TradeSignal | None:
    """Recompute executable signal fields from a fresh market response."""
    loc = LOCATIONS.get(signal.city_slug)
    if loc is None:
        return None

    contract = contract_matches_strategy(
        market,
        city_name=loc["name"],
        station=loc["station"],
        unit=signal.unit,
        date_str=signal.date_str,
    )
    if not contract.valid:
        logger.warning(
            "Live revalidation rejected %s: %s",
            signal.market_id,
            contract.reason,
        )
        return None

    quote = _token_quote_from_market(market, signal.token_id)
    if quote is None:
        logger.warning("Live revalidation found no quote for %s", signal.token_id)
        return None

    bid, ask = quote
    spread = ask - bid
    if ask >= MAX_PRICE or spread > MAX_SLIPPAGE:
        logger.info(
            "Live revalidation skipped %s: ask=%.4f spread=%.4f",
            signal.token_id,
            ask,
            spread,
        )
        return None

    fee_rate = market_fee_rate(market)
    ev = calc_ev(signal.probability, ask, fee_rate)
    if ev < MIN_EV:
        logger.info(
            "Live revalidation skipped %s: EV %.4f below %.4f",
            signal.token_id,
            ev,
            MIN_EV,
        )
        return None

    kelly_frac = calc_kelly(signal.probability, ask, fee_rate)
    size_usdc = min(bet_size(kelly_frac, balance_ref), MAX_BET)
    shares = round(size_usdc / ask, 2)
    if shares <= 0 or size_usdc <= 0:
        return None

    raw_ev = (
        calc_ev(signal.raw_probability, ask, fee_rate)
        if signal.raw_probability is not None
        else signal.raw_ev
    )
    return replace(
        signal,
        entry_price=ask,
        spread=spread,
        ev=ev,
        kelly=kelly_frac,
        size_usdc=size_usdc,
        shares=shares,
        fee_rate=fee_rate,
        raw_ev=raw_ev,
        reason=f"EV {ev:+.2f} @ ${ask:.3f}",
    )


# =============================================================================
# LIVE EXECUTION LAYER
# =============================================================================


class LiveExecutor:
    """Connects the weather strategy to the Polymarket CLOB.

    Lifecycle:
        executor = LiveExecutor(private_key, wallet)
        executor.connect()              # → auth + approvals
        executor.scan_and_execute()     # → scan → signal → execute
        executor.close()                # → cleanup
    """

    def __init__(
        self,
        private_key: str,
        wallet: str | None = None,
        *,
        gateway: Any | None = None,
        state: dict | None = None,
        state_saver: Callable[[dict], None] = save_live_state,
    ) -> None:
        self._private_key = private_key
        self._wallet = wallet
        self._client: SecureClient | None = None
        self._public: PublicClient | None = None
        self._gateway = gateway
        self._state_saver = state_saver
        self._state: dict = state if state is not None else load_live_state()
        self._cal = load_cal()
        self._forecast_cache = build_forecast_cache()
        self._consecutive_errors = 0
        self._circuit_open = False

    # ── Connection ─────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Authenticate with Polymarket and set up trading approvals."""
        logger.info("Connecting to Polymarket CLOB...")
        if not self._private_key:
            raise RuntimeError("No private key. Set PK in .env")

        self._client = SecureClient.create(
            private_key=self._private_key,
            wallet=self._wallet,
            environment=PRODUCTION,
        )
        wallet_type = self._client.wallet_type
        wallet_addr = self._client.wallet
        logger.info("Connected — wallet: %s (%s)", wallet_addr, wallet_type)

        # Public client for read-only queries
        self._public = PublicClient(environment=PRODUCTION)
        self._gateway = PolymarketGateway(self._client)

        # One-time trading approvals (USDC + CLOB)
        if self._state.get("first_run", True):
            logger.info("Setting up trading approvals (this runs once)...")
            self._client.setup_trading_approvals()
            self._state["first_run"] = False
            self._save_state()
            logger.info("Trading approvals complete")

    @property
    def client(self) -> SecureClient:
        if self._client is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._client

    @property
    def public(self) -> PublicClient:
        if self._public is None:
            self._public = PublicClient(environment=PRODUCTION)
        return self._public

    @property
    def gateway(self) -> Any:
        if self._gateway is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._gateway

    def _save_state(self) -> None:
        self._state_saver(self._state)

    def close(self) -> None:
        """Close all network transports."""
        if self._client:
            self._client.close()
            self._client = None
        if self._public:
            self._public.close()
            self._public = None

    # ── Balance ────────────────────────────────────────────────────────────

    def get_balance(self) -> Decimal:
        """Query on-chain USDC balance and allowance."""
        try:
            balance = self.gateway.get_balance()
            logger.info("Balance: $%s USDC", f"{balance:,.2f}")
            return balance
        except Exception as e:
            logger.warning("Balance check failed: %s", e)
        return Decimal("0")

    def reconcile_exchange_state(self) -> ReconciliationReport:
        """Make exchange-held positions authoritative over local estimates."""
        snapshots = self.gateway.get_positions()
        report = reconcile_state(self._state, snapshots)
        self._save_state()
        if report.has_discrepancies:
            self._circuit_open = True
            logger.error(
                "Exchange reconciliation discrepancy: unmanaged=%s missing=%s",
                report.unmanaged_tokens,
                report.missing_tokens,
            )
        return report

    # ── Core Scan Loop ─────────────────────────────────────────────────────

    def scan_and_execute(self) -> int:
        """Run one full strategy scan and execute any trade signals.

        Returns:
            Number of orders placed this cycle.
        """
        if self._circuit_open:
            logger.warning("Circuit breaker is OPEN — skipping scan")
            send_telegram("⚠️ Circuit breaker open. Restart after manual review.")
            return 0

        logger.info("=" * 60)
        logger.info("SCAN CYCLE START")
        placed = 0

        try:
            # 1. Reconcile local state before making any trading decision.
            report = self.reconcile_exchange_state()
            if report.has_discrepancies:
                send_telegram(
                    "🚨 Exchange/local position mismatch. Trading halted for review."
                )
                return 0

            # 2. Check on-chain balance
            onchain_balance = self.get_balance()
            if onchain_balance < Decimal("1.0"):
                logger.warning(
                    "On-chain balance too low: %s USDC", onchain_balance
                )
                send_telegram(
                    f"⚠️ Low balance: {onchain_balance:.2f} USDC. Skipping scan."
                )
                self._consecutive_errors += 1
                return 0

            # 3. Refresh approvals
            self.client.setup_trading_approvals()

            # 4. Load calibration
            self._cal = load_cal()

            # 5. Fetch outdoor markets
            markets = fetch_outdoor_markets()
            if not markets:
                logger.warning("No markets returned from Gamma")
                self._consecutive_errors += 1
                return 0

            # 6. Generate trade signals
            signals = generate_signals(
                markets,
                self._state,
                forecast_cache=self._forecast_cache,
            )

            # 7. Execute best signals (cap by balance)
            max_spend = min(
                float(onchain_balance) * KELLY_FRACTION,
                MAX_BET * 3,  # max 3 simultaneous entries per cycle
            )
            total_spend = 0.0

            for signal in signals:
                if total_spend + signal.size_usdc > max_spend:
                    logger.info(
                        "Budget limit reached (%.2f / %.2f)",
                        total_spend,
                        max_spend,
                    )
                    break
                if self._execute_signal(signal):
                    total_spend += signal.size_usdc
                    placed += 1

            # 8. Check open positions for exit conditions
            self._check_positions()

            # 9. Update calibration
            try:
                if self._state.get("total_trades", 0) >= weatherbet.CALIBRATION_MIN:
                    all_mkts = weatherbet.load_all_markets()
                    resolved = [
                        m
                        for m in all_mkts
                        if m.get("resolved") and m.get("actual_temp") is not None
                    ]
                    if len(resolved) >= weatherbet.CALIBRATION_MIN:
                        cal = weatherbet.run_calibration(all_mkts)
                        self._cal = cal
            except Exception as e:
                logger.debug("Calibration update skipped: %s", e)

            self._consecutive_errors = 0

        except ConnectionError as e:
            self._consecutive_errors += 1
            logger.error("Connection error: %s", e)
            if self._consecutive_errors >= 5:
                self._circuit_open = True
                send_telegram(
                    f"🚨 Circuit breaker OPEN after {self._consecutive_errors}"
                    f" consecutive errors. Manual restart required."
                )
        except Exception as e:
            self._consecutive_errors += 1
            logger.error(
                "Scan failed (%dx): %s", self._consecutive_errors, e,
                exc_info=True,
            )
            if self._consecutive_errors >= 5:
                self._circuit_open = True
                send_telegram(
                    f"🚨 Circuit breaker OPEN after {self._consecutive_errors}"
                    f" consecutive errors: {e}"
                )

        logger.info("SCAN CYCLE END — %d orders placed", placed)
        return placed

    # ── Signal Execution ───────────────────────────────────────────────────

    def _execute_signal(self, signal: TradeSignal) -> bool:
        """Place a live order from a trade signal.

        Returns True on successful order placement.
        """
        risk = assess_trade_risk(
            self._state,
            size_usdc=signal.size_usdc,
            city_slug=signal.city_slug,
            date_str=signal.date_str,
            signal_created_at=signal.created_at_ts,
            bankroll=float(self._state.get("balance_ref", BALANCE_REF)),
            limits=RISK_LIMITS,
            now_ts=time.time(),
        )
        if not risk.allowed:
            logger.warning(
                "Risk gate rejected %s %s: %s (total=%.2f event=%.2f daily_loss=%.2f active=%d)",
                signal.city_slug,
                signal.date_str,
                risk.reason,
                risk.total_exposure,
                risk.event_exposure,
                risk.daily_loss,
                risk.active_positions,
            )
            return False

        detail = fetch_market_detail(signal.market_id)
        if detail is None:
            logger.warning(
                "Skipping %s: fresh market detail unavailable before order",
                signal.token_id,
            )
            return False

        refreshed_signal = refresh_signal_with_live_market(
            signal,
            detail,
            balance_ref=float(self._state.get("balance_ref", BALANCE_REF)),
        )
        if refreshed_signal is None:
            return False
        signal = refreshed_signal

        risk = assess_trade_risk(
            self._state,
            size_usdc=signal.size_usdc,
            city_slug=signal.city_slug,
            date_str=signal.date_str,
            signal_created_at=signal.created_at_ts,
            bankroll=float(self._state.get("balance_ref", BALANCE_REF)),
            limits=RISK_LIMITS,
            now_ts=time.time(),
        )
        if not risk.allowed:
            logger.warning(
                "Risk gate rejected refreshed %s %s: %s (total=%.2f event=%.2f daily_loss=%.2f active=%d)",
                signal.city_slug,
                signal.date_str,
                risk.reason,
                risk.total_exposure,
                risk.event_exposure,
                risk.daily_loss,
                risk.active_positions,
            )
            return False

        bucket_label = (
            f"{signal.bucket_low}-{signal.bucket_high}{signal.unit}"
        )
        logger.info(
            "➡ Signal: BUY %s %s | %s | prob=%.1f%% EV=%.2f | "
            "$%.2f @ $%.3f (%s)",
            signal.city_name,
            signal.date_str,
            bucket_label,
            signal.probability * 100,
            signal.ev,
            signal.size_usdc,
            signal.entry_price,
            signal.forecast_source.upper(),
        )

        # Persist intent before the network call. If the response is lost, the
        # token remains blocked until exchange reconciliation establishes truth.
        pos = {
            "market_id": signal.market_id,
            "token_id": signal.token_id,
            "condition_id": signal.condition_id,
            "order_id": None,
            "side": "BUY",
            "city_slug": signal.city_slug,
            "city_name": signal.city_name,
            "date": signal.date_str,
            "bucket_low": signal.bucket_low,
            "bucket_high": signal.bucket_high,
            "unit": signal.unit,
            "forecast_temp": signal.forecast_temp,
            "raw_forecast_temp": signal.raw_forecast_temp,
            "corrected_forecast_temp": signal.corrected_forecast_temp,
            "forecast_bias": signal.forecast_bias,
            "forecast_raw_bias": signal.forecast_raw_bias,
            "forecast_lead_bucket": signal.forecast_lead_bucket,
            "forecast_calibration_n": signal.forecast_calibration_n,
            "forecast_source": signal.forecast_source,
            "sigma": signal.sigma,
            "fee_rate": signal.fee_rate,
            "probability": signal.probability,
            "raw_probability": signal.raw_probability,
            "ev": signal.ev,
            "raw_ev": signal.raw_ev,
            "requested_amount": signal.size_usdc,
            "amount": 0.0,
            "shares": 0.0,
            "entry_price": signal.entry_price,
            "entry_bid": round(signal.entry_price - signal.spread, 4),
            "spread": signal.spread,
            "signal_created_at": signal.created_at_ts,
            "status": "submitting",
            "stop_price": round(signal.entry_price * 0.80, 4),
            "trailing_activated": False,
            "entered_at": datetime.now(timezone.utc).isoformat(),
            "exited_at": None,
            "exit_price": None,
            "pnl": None,
            "close_reason": None,
        }
        self._state["positions"].append(pos)
        self._save_state()

        try:
            submission = self.gateway.buy(
                token_id=signal.token_id,
                amount=Decimal(str(signal.size_usdc)),
                max_price=Decimal(str(signal.entry_price)),
            )
        except Exception as error:
            pos["status"] = "unknown"
            pos["submission_error"] = str(error)
            self._save_state()
            logger.error(
                "BUY outcome unknown for %s; reconciliation required: %s",
                signal.token_id,
                error,
            )
            return False

        if not submission.accepted:
            pos["status"] = "rejected"
            pos["submission_error"] = submission.reason
            self._state["failed_signals"].append(
                {
                    "token_id": signal.token_id,
                    "reason": submission.reason or "rejected",
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
            )
            self._save_state()
            logger.warning("Order rejected for %s: %s", signal.token_id, submission.reason)
            return False

        pos["order_id"] = submission.order_id
        pos["order_status"] = submission.status
        pos["status"] = "pending"
        snapshot = self.gateway.get_positions().get(signal.token_id)
        reconcile_entry(pos, snapshot)
        if snapshot is not None:
            self._state["total_trades"] = self._state.get("total_trades", 0) + 1
        self._save_state()

        msg = (
            f"✅ BUY {signal.city_name} {signal.date_str} "
            f"{bucket_label} | EV {signal.ev:+.2f} | "
            f"requested ${signal.size_usdc:.2f} @ max ${signal.entry_price:.3f} | "
            f"{submission.status} | local={pos['status']}"
        )
        logger.info(msg)
        send_telegram(msg)
        return True

    # ── Position Management ────────────────────────────────────────────────

    def _check_positions(self) -> None:
        """Check all open positions for exit conditions.

        Exits:
            - Stop-loss: price drops >20% from entry
            - Take-profit: price reaches 0.85 (near resolution) or 0.75 (early)
            - Trailing stop: price up 20%+ → move stop to breakeven
        """
        closed = 0

        try:
            self.reconcile_exchange_state()
        except Exception as error:
            logger.error("Position reconciliation failed: %s", error)
            return

        self._redeem_resolved_positions()

        for pos in self._state.get("positions", []):
            if pos.get("status") != "open":
                continue

            try:
                # Get current price from Gamma API
                detail = fetch_market_detail(pos["market_id"])
                if not detail:
                    continue

                current_price = float(
                    self.gateway.get_executable_sell_price(
                        token_id=pos["token_id"],
                        shares=Decimal(str(pos.get("shares", 0))),
                    )
                )

                entry = float(pos["entry_price"])
                stop = float(pos.get("stop_price", entry * 0.80))

                # Hours to resolution
                end_dt = detail.get("endDate", "")
                hrs_left = hours_to_resolution(end_dt)

                # ── Trailing stop logic ─────────────────────────
                if not pos.get("trailing_activated") and current_price >= entry * 1.20:
                    pos["stop_price"] = round(entry, 4)
                    pos["trailing_activated"] = True
                    logger.info(
                        "🔒 Trailing: %s %s — stop moved to breakeven $%.3f",
                        pos["city_name"],
                        pos["date"],
                        entry,
                    )

                # ── Take-profit threshold ───────────────────────
                if hrs_left < 24:
                    take_profit_price = None  # hold to resolution
                elif hrs_left < 48:
                    take_profit_price = 0.85
                else:
                    take_profit_price = 0.75

                # ── Exit conditions ─────────────────────────────
                if take_profit_price is not None and current_price >= take_profit_price:
                    if self._exit_position(pos, current_price, "take_profit"):
                        closed += 1
                elif current_price <= stop:
                    if current_price < entry:
                        did_close = self._exit_position(pos, current_price, "stop_loss")
                    else:
                        did_close = self._exit_position(pos, current_price, "trailing_stop")
                    if did_close:
                        closed += 1

            except Exception as e:
                logger.debug(
                    "Position check error for %s: %s",
                    pos.get("market_id"),
                    e,
                )
                continue

        if closed:
            self._save_state()
            logger.info("Closed %d positions this cycle", closed)

    def _redeem_resolved_positions(self) -> None:
        for pos in self._state.get("positions", []):
            if pos.get("status") != "open" or not pos.get("redeemable"):
                continue
            self._redeem_position(pos)

    def _redeem_position(self, pos: dict) -> bool:
        condition_id = str(pos.get("condition_id", ""))
        if not condition_id:
            logger.error("Cannot redeem %s without condition_id", pos.get("token_id"))
            return False

        pos["status"] = "redeeming"
        self._save_state()
        try:
            transaction_hash = self.gateway.redeem(condition_id=condition_id)
        except Exception as error:
            pos["status"] = "redemption_unknown"
            pos["redemption_error"] = str(error)
            self._save_state()
            logger.error("Redemption outcome unknown for %s: %s", condition_id, error)
            return False

        shares = float(pos.get("shares", 0))
        entry_cost = float(pos.get("amount", 0))
        won = float(pos.get("current_price", 0)) >= 0.5
        payout = shares if won else 0.0
        pnl = round(payout - entry_cost, 2)
        pos["status"] = "redemption_confirmed"
        pos["redemption_tx"] = transaction_hash
        pos["close_reason"] = "resolved"
        pos["exit_price"] = 1.0 if won else 0.0
        pos["pnl"] = pnl
        pos["exited_at"] = datetime.now(timezone.utc).isoformat()
        if won:
            self._state["wins"] = self._state.get("wins", 0) + 1
        else:
            self._state["losses"] = self._state.get("losses", 0) + 1
        self._save_state()
        logger.info("Redemption confirmed for %s: %s", condition_id, transaction_hash)
        return True

    def _exit_position(
        self,
        pos: dict,
        current_price: float,
        reason: str,
    ) -> bool:
        """Sell shares to close a position."""
        token_id = pos["token_id"]
        shares = Decimal(str(pos.get("shares", 0)))
        min_price = Decimal(str(current_price * 0.95))

        if shares <= 0:
            logger.error("Refusing exit with no confirmed shares for %s", token_id)
            return False

        logger.info(
            "Exit %s %s: %s (price=$%.3f shares=%s)",
            pos["city_name"],
            pos["date"],
            reason,
            current_price,
            shares,
        )

        pos["status"] = "exit_pending"
        pos["shares_before_exit"] = float(shares)
        pos["exit_price_requested"] = current_price
        pos["close_reason"] = reason
        self._save_state()

        try:
            submission = self.gateway.sell(
                token_id=token_id,
                shares=shares,
                min_price=min_price,
            )
        except Exception as error:
            pos["status"] = "exit_unknown"
            pos["exit_error"] = str(error)
            self._save_state()
            logger.error(
                "Exit outcome unknown for %s: %s", token_id, error,
                exc_info=True,
            )
            return False

        if not submission.accepted:
            pos["status"] = "open"
            pos["exit_error"] = submission.reason
            self._save_state()
            logger.warning("Exit rejected for %s: %s", token_id, submission.reason)
            return False

        pos["exit_order_id"] = submission.order_id
        pos["exit_order_status"] = submission.status
        snapshot = self.gateway.get_positions().get(token_id)
        if snapshot is None:
            pos["exit_absence_confirmations"] = 1
            closed = False
        elif snapshot.shares < shares:
            closed = reconcile_exit(
                pos,
                snapshot,
                exit_price=Decimal(str(current_price)),
            )
        else:
            closed = False
        pos["exit_price"] = round(current_price, 4)
        if closed:
            pos["exited_at"] = datetime.now(timezone.utc).isoformat()
        self._save_state()

        if not closed:
            logger.warning(
                "Partial or unconfirmed exit for %s; residual shares=%s",
                token_id,
                pos.get("shares"),
            )
            return False

        pnl = float(pos.get("pnl", 0))

        if pnl >= 0:
            self._state["wins"] = self._state.get("wins", 0) + 1
            outcome = "WIN"
        else:
            self._state["losses"] = self._state.get("losses", 0) + 1
            outcome = "LOSS"

        self._save_state()

        bucket_label = (
            f"{pos['bucket_low']}-{pos['bucket_high']}{pos['unit']}"
        )
        msg = (
            f"{'🔴' if pnl < 0 else '🟢'} {outcome} {pos['city_name']} "
            f"{pos['date']} {bucket_label} | "
            f"exit ${current_price:.3f} | "
            f"PnL: {'+' if pnl >= 0 else ''}{pnl:.2f} | "
            f"{reason}"
        )
        logger.info(msg)
        send_telegram(msg)
        return True

    # ── Utility ────────────────────────────────────────────────────────────

    def cancel_all_orders(self) -> int:
        """Cancel all open orders on the CLOB.

        Returns:
            Number of cancelled orders.
        """
        try:
            resp = self.client.cancel_all()
            count = len(resp.cancelled_orders) if hasattr(resp, "cancelled_orders") else 0
            logger.info("Cancelled %d orders", count)
            return count
        except Exception as e:
            logger.error("Cancel all failed: %s", e)
            return 0

    def status_report(self) -> str:
        """Generate a status report string for the terminal or Telegram."""
        onchain_balance = self.get_balance()
        positions = [
            p for p in self._state.get("positions", [])
            if p.get("status")
            in (
                "submitting",
                "pending",
                "unknown",
                "open",
                "exit_pending",
                "exit_unknown",
                "redeeming",
                "redemption_unknown",
                "redemption_confirmed",
                "missing",
                "unmanaged",
            )
        ]
        closed = [
            p for p in self._state.get("positions", [])
            if p.get("status") == "closed"
        ]
        total_trades = self._state.get("total_trades", 0)
        wins = self._state.get("wins", 0)
        losses = self._state.get("losses", 0)

        lines = [
            "=" * 55,
            "  WEATHERBET — LIVE STATUS",
            "=" * 55,
            f"  On-chain balance: ${float(onchain_balance):,.2f}",
            f"  Trades: {total_trades} | W: {wins} | L: {losses}",
        ]
        if total_trades:
            lines.append(f"  Win rate: {wins/total_trades:.0%}")
        lines.append(f"  Open positions: {len(positions)}")
        lines.append(f"  Circuit breaker: {'OPEN ⚠️' if self._circuit_open else 'CLOSED ✓'}")

        if positions:
            lines.append("")
            lines.append("  Open positions:")
            for p in positions:
                if p.get("status") == "unmanaged" or not p.get("city_name"):
                    lines.append(
                        f"    {p.get('status', 'unknown').upper():<16} "
                        f"{p.get('token_id', 'unknown')} | "
                        f"shares {p.get('shares', 0)}"
                    )
                    continue
                bucket = f"{p['bucket_low']}-{p['bucket_high']}{p['unit']}"
                lines.append(
                    f"    {p['city_name']:<16} {p['date']} | "
                    f"{bucket:<14} | entry ${p['entry_price']:.3f} | "
                    f"{p['forecast_source'].upper()}"
                )

        lines.append("=" * 55)
        return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================


def cmd_loop(executor: LiveExecutor) -> None:
    """Run the live trading loop."""
    executor.connect()

    logger.info(
        "Starting live trading loop (scan every %ds)",
        SCAN_INTERVAL,
    )
    send_telegram("🌤️ Weatherbet live trading started")

    last_full_scan = 0
    try:
        while True:
            now_ts = time.time()
            if now_ts - last_full_scan >= SCAN_INTERVAL:
                executor.scan_and_execute()
                last_full_scan = now_ts
            else:
                # Quick position check mid-cycle
                executor._check_positions()
                executor._save_state()
                time.sleep(60)  # check positions every 60s
    except KeyboardInterrupt:
        logger.info("Stopping — saving state...")
        executor._save_state()
        executor.close()
        logger.info("Done. Bye!")
    except Exception as e:
        logger.critical("Fatal error: %s", e, exc_info=True)
        send_telegram(f"🚨 Bot crashed: {e}")
        executor._save_state()
        executor.close()
        raise


def main() -> None:
    """CLI entry point."""
    if len(sys.argv) > 1:
        command = sys.argv[1]
    else:
        command = "run"

    if not PRIVATE_KEY:
        print("Error: PK environment variable is required.")
        print("Set PK=0x... in .env file.")
        sys.exit(1)

    if command not in {"run", "status", "cancel", "scan"}:
        print(f"Usage: {sys.argv[0]} [run|status|cancel|scan]")
        print("  run      (default) Live trading loop")
        print("  status   Show account and position status")
        print("  cancel   Cancel all open CLOB orders")
        print("  scan     Run one scan cycle and exit")
        sys.exit(1)

    try:
        with ProcessLock(LIVE_LOCK_FILE):
            executor = LiveExecutor(private_key=PRIVATE_KEY, wallet=WALLET_ADDR)
            if command == "run":
                cmd_loop(executor)
            elif command == "status":
                try:
                    executor.connect()
                    print(executor.status_report())
                finally:
                    executor.close()
            elif command == "cancel":
                try:
                    executor.connect()
                    count = executor.cancel_all_orders()
                    print(f"Cancelled {count} orders")
                finally:
                    executor.close()
            elif command == "scan":
                try:
                    executor.connect()
                    placed = executor.scan_and_execute()
                    print(f"Placed {placed} orders this cycle")
                finally:
                    executor.close()
    except ProcessLockError as error:
        logger.error("%s", error)
        sys.exit(2)


if __name__ == "__main__":
    main()
