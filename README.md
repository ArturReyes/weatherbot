# 🌤 WeatherBet — Polymarket Weather Trading Bot

Automated weather market trading bot for Polymarket. Finds mispriced temperature outcomes using real forecast data from multiple sources across 20 cities worldwide.

Paper trading is handled by `weatherbet.py`. Live order execution is handled separately by `live_executor.py`.

---

## Versions

### `bot_v1.py` — Base Bot
The foundation. Scans 6 US cities, fetches forecasts from NWS using airport station coordinates, finds matching temperature buckets on Polymarket, and enters trades when the market price is below the entry threshold.

No math, no complexity. Just the core logic — good for understanding how the system works.

### `weatherbet.py` — Full Bot (current)
Everything in v1, plus:
- **20 cities** across 4 continents (US, Europe, Asia, South America, Oceania)
- **3 forecast sources** — ECMWF (global), real NOAA HRRR Conus (US, hourly), METAR (real-time observations)
- **Expected Value** — skips trades where the math doesn't work
- **Kelly Criterion** — sizes positions based on edge strength
- **Strategy-aware exits** — calibrated positions follow forecast invalidation/resolution; optional price stops use the executable entry bid
- **Spread filters** — rejects both wide absolute spreads and disproportionate spreads on cheap buckets
- **Self-calibration** — learns sigma and bias by city/source/lead-time bucket
- **Bias correction** — applies capped forecast corrections before probability calculation
- **Structural edge strategies** — calibrated mean, near-lock, underdispersion tails, model-lag, and gated NO-token entries
- **Full data storage** — every forecast snapshot, trade, and resolution saved to JSON
- **Eventual-resolution tracking** — early paper exits keep their original P&L while the market is followed to settlement for Brier scoring

### `live_executor.py` — Live Trading
Connects `weatherbet.py`'s strategy engine to the Polymarket CLOB via `polymarket-client`. Full live trading loop with:
- FAK market-order entry/exit through the SDK
- On-chain USDC balance checks
- Fresh market revalidation before order submission
- Portfolio risk gates and exposure limits
- Shared strategy-aware take-profit and optional spread-safe stop exits
- Strategy-aware exits (near-lock positions hold to resolution unless invalidated by METAR)
- Circuit breaker (stops after 5 consecutive errors)
- Telegram notifications
- Atomic state file writes (no corruption on crash)

---

## How It Works

Polymarket runs markets like "Will the highest temperature in Chicago be between 46–47°F on March 7?" These markets are often mispriced — the forecast says 78% likely but the market is trading at 8 cents.

The bot:
1. Fetches forecasts from ECMWF and real HRRR Conus via Open-Meteo (free, no key required)
2. Gets real-time observations from METAR airport stations
3. Applies city/source/lead-time calibration and capped bias correction
4. Generates ranked strategy candidates instead of only checking the bucket containing the mean
5. Scores calibrated-mean, near-lock, underdispersion-tail, model-lag, and gated NO-token opportunities by fee-adjusted EV
6. Sizes the position using fractional Kelly Criterion and strategy multipliers
7. Monitors exits with strategy-aware rules
8. Auto-resolves markets by querying Polymarket API directly

---

## Why Airport Coordinates Matter

Most bots use city center coordinates. That's wrong.

Every Polymarket weather market resolves on a specific airport station. NYC resolves on LaGuardia (KLGA), Dallas on Love Field (KDAL), and Paris on Le Bourget (LFPB). The difference between city center and airport can be 3–8°F. On markets with 1–2°F buckets, that's the difference between the right trade and a guaranteed loss.

| City | Station | Airport |
|------|---------|---------|
| NYC | KLGA | LaGuardia |
| Chicago | KORD | O'Hare |
| Miami | KMIA | Miami Intl |
| Dallas | KDAL | Love Field |
| Seattle | KSEA | Sea-Tac |
| Atlanta | KATL | Hartsfield |
| London | EGLC | London City |
| Tokyo | RJTT | Haneda |
| ... | ... | ... |

---

## Installation

```bash
git clone https://github.com/alteregoeth-ai/weatherbot
cd weatherbot
pip install -r requirements.txt

# For live trading (optional — only if using live_executor.py):
pip install --pre polymarket-client

# Copy and edit your secrets:
cp .env.example .env.local
```

> `.env` and `.env.local` are in `.gitignore` — your secrets stay local. Never commit either file.

---

## Usage and Safe Progression

### 1. Paper trade first

This is the safe starting mode. It writes simulated positions and calibration data under `data/`, and mirrors the operator console to `paper_trading.log`.

```bash
python weatherbet.py run       # start paper trading loop
python weatherbet.py status    # balance and open positions
python weatherbet.py report    # full breakdown of all resolved markets
python weatherbet.py validate  # chronological paper holdout + live-readiness gates
python weatherbet.py archive-reset  # archive a finished cohort and reset paper bankroll/counters
python weatherbet.py repair-calibration --dry-run  # inspect recoverable Polymarket settlements
python weatherbet.py repair-calibration --apply    # annotate history and rebuild calibration
```

`paper_trading.log` is plain text with ANSI color codes stripped, so it is safe to paste into reviews or inspect with `tail -f paper_trading.log`.

You can also run `python weatherbet.py` with no argument; it defaults to `run`.

`archive-reset` fails closed if any paper position is open. It writes a position-level snapshot under `data/evaluations/`, marks legacy trades so later settlement cannot pollute the new counters, resets the configured paper bankroll, and advances the evaluation timestamp. Forecast snapshots, actual temperatures, market history, and `data/calibration.json` are retained.

### 2. Live account status only

This connects to the exchange and reads account/position state. It should not place orders.

```bash
python live_executor.py status
```

### 3. Live trading commands

These commands can change real exchange state.

```bash
python live_executor.py run            # live trading loop; can place and exit orders
python live_executor.py                # same as run
python live_executor.py scan           # one live scan cycle; can place orders
python live_executor.py cancel         # cancels all open CLOB orders
```

Important: `live_executor.py scan` is not a dry run. If your `.env` has a valid funded `PK`, it can submit live FAK orders.

Recommended order:

1. Run `weatherbet.py run` in paper mode.
2. Review `weatherbet.py status` and `weatherbet.py report`.
3. Confirm calibration files are accumulating in `data/`.
4. Add a true no-order dry-run mode before using live execution for signal testing.
5. Only then use `live_executor.py run` or `live_executor.py scan` with real funds.

---

## What API Keys Do You Need?

| Key | Required? | Where to get it |
|-----|-----------|-----------------|
| `PK` | Only for live trading | Your MetaMask / wallet's Polygon private key (see below) |
| `VC_KEY` | No; diagnostics only | https://www.visualcrossing.com/weather-api (free) |
| `TELEGRAM_BOT_TOKEN` | No | https://t.me/BotFather |
| `TELEGRAM_CHAT_ID` | No | Get from Telegram API (see below) |

### 1. `PK` — Polymarket Wallet Private Key

This is your Polygon EOA private key. The bot signs orders with it.

**If you have MetaMask:**

```
MetaMask → three dots → Account details → Show private key → copy 0x...
```

**If you don't have a wallet yet:**

1. Install MetaMask: https://metamask.io
2. Create a new wallet — save the seed phrase
3. Add Polygon network:
   - Network Name: Polygon
   - RPC URL: https://polygon-rpc.com
   - Chain ID: 137
4. Fund it: buy USDC.e on Polygon or bridge via https://wallet.polygon.technology/
5. Export private key from MetaMask as above

> ⚠️ This key controls your funds. Never share it, never commit it, never type it where anyone can see.

### 2. `VC_KEY` — Visual Crossing (optional diagnostics)

Optional. The paper scanner may retain Visual Crossing daily highs for comparison, but these readings never drive calibration. Calibration is derived from Polymarket's unique settled winning bucket because that is the outcome the contracts pay against.

```bash
# 1. Sign up: https://www.visualcrossing.com/weather-api
# 2. Free tier: 1,000 queries/day
# 3. Dashboard → API Keys → copy
```

> Missing `VC_KEY` does not block calibration. Open-ended winning buckets are excluded because they do not identify a precise temperature.

### 3. Telegram (for push notifications)

```bash
# Talk to @BotFather: /newbot → name it → copy the bot token
# Send any message to your new bot
# Visit: https://api.telegram.org/bot<TOKEN>/getUpdates
# Copy the "chat": { "id": 123456789 }
```

---

## Configure `.env`

Paper trading does not require a private key. Live trading does.

```env
# === REQUIRED FOR LIVE TRADING ONLY ===
PK=0xabc123def456...your_private_key_here

# === OPTIONAL ===
WALLET=0x...            # wallet address (derived from PK if omitted)
VC_KEY=your_key_here    # Visual Crossing (calibration)
TELEGRAM_BOT_TOKEN=     # for notifications
TELEGRAM_CHAT_ID=       # your Telegram user/group ID
```

---

## Strategy Parameters (`config.json`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `balance` | 10000 | Starting balance for Kelly sizing |
| `max_bet` | 20 | Max bet per trade in USDC |
| `min_ev` | 0.10 | Minimum expected value required |
| `max_price` | 0.45 | Max entry price (avoid expensive YES shares) |
| `max_slippage` | 0.03 | Max allowed absolute ask-bid spread |
| `max_relative_spread` | 0.25 | Reject when spread exceeds 25% of the ask, especially on cheap buckets |
| `min_trade_notional` | 0.50 | Operator floor; the current CLOB minimum-share cost can raise the effective minimum |
| `kelly_fraction` | 0.25 | Fraction of Kelly to bet (0.25 = quarter-Kelly) |
| `scan_interval` | 3600 | Seconds between full market scans |
| `opportunity_scan_interval_seconds` | 300 | Faster structural-edge scan cadence; capped by `scan_interval` |
| `calibration_min` | 30 | Samples required for a calibration entry to be marked mature |
| `calibration_bootstrap_min` | 7 | Begin shrinkage-regularized calibration from settled Polymarket winning buckets |
| `calibrated_mean_min_samples` | 7 | Paper calibrated-mean bootstrap gate; aggregate city/source calibration is used until a lead bucket matures |
| `live_calibrated_mean_min_samples` | 30 | Independent live signal maturity gate; live readiness checks still apply |
| `max_total_exposure_pct` | 0.25 | Max total portfolio exposure as share of bankroll |
| `max_event_exposure_pct` | 0.10 | Max same city/date exposure as share of bankroll |
| `max_daily_loss_pct` | 0.05 | Stops new entries after daily realized loss breach |
| `max_open_positions` | 5 | Max active positions |
| `max_signal_age_seconds` | 120 | Rejects stale signals before live order submission |
| `forecast_cache_ttl_ecmwf_seconds` | 1800 | ECMWF forecast cache TTL |
| `forecast_cache_ttl_hrrr_seconds` | 600 | Real HRRR Conus forecast cache TTL (US only) |
| `forecast_cache_ttl_metar_seconds` | 45 | METAR observation cache TTL |
| `weather_api_user_agent` | `WeatherBet/1.0 weather-trading-operator` | Identifies the client to weather providers; can be overridden by `WEATHER_API_USER_AGENT` |
| `weather_api_max_attempts` | 3 | Bounded attempts for DNS, connection, timeout, `429`, and upstream `5xx` failures |
| `weather_api_retry_base_seconds` | 0.75 | Initial exponential-backoff delay before jitter |
| `weather_api_retry_max_seconds` | 6.0 | Maximum delay for one weather API retry, including `Retry-After` |
| `max_bias_correction_f` | 3.0 | Max forecast bias correction in °F |
| `max_bias_correction_c` | 1.5 | Max forecast bias correction in °C |
| `strategy_calibrated_mean_enabled` | true | Enable existing corrected-forecast bucket strategy |
| `strategy_near_lock_enabled` | false | Enable D+0 near-lock only after continuous local-day METAR coverage |
| `strategy_underdispersion_enabled` | false | Reserve for a future true-ensemble feed; source disagreement is not ensemble spread |
| `strategy_model_lag_enabled` | false | Require both a forecast-probability move and insufficient bucket-specific market repricing |
| `enable_no_trades` | false | Disabled by default; when enabled, entries require the NO token's own CLOB book quote |
| `standard_price_stop_enabled` | false | Keep calibrated weather exposure through bid noise; forecast invalidation remains active |
| `paper_reentry_enabled` | true | Permit a new paper entry after the cooldown while preserving prior trade history |
| `paper_reentry_cooldown_minutes` | 60 | Minimum wait before paper re-entry |
| `paper_max_entries_per_market` | 2 | Maximum paper entries for one token market |
| `live_reentry_enabled` | false | Live repeated entry remains fail-closed until explicitly enabled |
| `near_lock_hours` | 18 | Hours-to-resolution window for near-lock |
| `near_lock_min_prob` | 0.92 | Minimum near-lock probability |
| `near_lock_max_price` | 0.82 | Strategy-specific max entry price for near-lock |
| `near_lock_sigma_f` | 0.75 | Conservative near-lock uncertainty in °F |
| `near_lock_sigma_c` | 0.4 | Conservative near-lock uncertainty in °C |
| `underdispersion_ratio_min` | 1.6 | Minimum calibrated-sigma / true-ensemble-spread ratio when an ensemble feed is added |
| `underdispersion_tail_max_price` | 0.14 | Max YES price for underdispersion tail entries |
| `model_lag_max_reprice_ratio` | 0.5 | Market may absorb at most this fraction of the forecast probability move |
| `validation_holdout_fraction` | 0.25 | Latest chronological share of closed paper trades reserved for holdout |
| `promotion_min_holdout_trades` | 30 | Required holdout trades per strategy before live eligibility |
| `promotion_min_brier_samples` | 30 | Required resolved probability samples per strategy |
| `promotion_min_realized_roi` | 0.0 | Minimum realised ROI on the holdout set |
| `promotion_max_brier_score` | 0.25 | Maximum holdout Brier score |
| `promotion_max_drawdown_pct` | 0.10 | Maximum holdout drawdown as share of paper bankroll |
| `require_validation_for_live` | true | Block live scans until every enabled strategy passes the chronological promotion gates |
| `no_trade_min_ev` | 0.15 | Stricter EV gate for NO-token entries |

---

## Pre-Flight Checklist

```bash
# 1. Install everything
pip install -r requirements.txt

# 2. Set up environment file
cp .env.example .env.local
# Paper trading requires no wallet key; VC_KEY is optional diagnostic data.
# For live trading, set PK and optional WALLET/Telegram values.

# 3. Verify imports
python -c "from weatherbet import bucket_prob; print('weatherbet OK')"

# 4. Paper trade for several days and collect calibration data
python weatherbet.py run
tail -f paper_trading.log
python weatherbet.py status
python weatherbet.py report

# 5. Only after paper validation, verify live account state
python live_executor.py status

# 6. Go live with small limits first
python live_executor.py
```

Do not use `python live_executor.py scan` as a dry run. It can place orders.

---

## Data Storage

All data is saved to `data/markets/` — one JSON file per market. Each file contains:
- Hourly forecast snapshots (ECMWF, HRRR, METAR)
- Market price history
- Position details (strategy, YES/NO side, entry, stop, PnL, raw/corrected forecast, bias, raw/corrected EV)
- Structural-edge diagnostics (fair price, edge, observed high, remaining max, dispersion ratio, source spread, model-lag shift)
- Exchange sizing evidence (`min_order_size`, required/proposed notional, sizing decision)
- Diagnostic shadow signals for otherwise valid opportunities rejected only by sizing
- Final resolution outcome

Calibration is saved to `data/calibration.json`. Only validated Polymarket winning buckets contribute observations: exact buckets use their value, bounded ranges use their midpoint, and open-ended tails are skipped. Raw third-party temperatures remain separate audit fields. The bot writes aggregate sigma entries and city/source/lead-bucket entries containing bias, raw bias, sigma, and sample count.

Before restarting after upgrading an existing dataset, run the repair command in dry-run mode, review its counts, and then apply it. Apply mode archives the prior calibration file and writes a repair audit under `data/evaluations/`. Shadow metrics appear in `status`, `report`, and `validate`, but never count as trades, ROI, drawdown, or promotion evidence.

Archived paper cohorts are saved under `data/evaluations/`. These archives preserve legacy state and position records for audit purposes without allowing old results to count toward the current promotion cohort.

---

## APIs Used

| API | Auth | Purpose |
|-----|------|---------|
| Open-Meteo | None | ECMWF + HRRR forecasts |
| Aviation Weather (METAR) | None | Real-time station observations |
| Polymarket Gamma | None | Market data |
| Polymarket CLOB | Wallet signature | Order placement (live trading only) |
| Visual Crossing | Free key | Optional historical-temperature diagnostics |

Weather provider calls fail closed: an unavailable or malformed response cannot
become a forecast or a trade signal. Open-Meteo and Aviation Weather requests use
bounded exponential backoff with jitter. HTTP `204` is treated as no observation,
`429` honors a bounded `Retry-After`, and retryable `5xx` responses are retried.
METAR observations for every configured station are fetched in one request and
cached briefly, avoiding a burst of one request per city.

---

## Security Notes

- All secrets go through `.env` only — never hardcoded in source files
- `.env` is in `.gitignore` — cannot be accidentally committed
- The `VC_KEY` was historically in `config.json` — **removed** in this version
- Private key (`PK`) never touches disk outside `.env`
- State writes use atomic file operations (`tempfile` + `os.replace`) — no corruption on crash

---

## Disclaimer

This is not financial advice. Prediction markets carry real risk. Run the simulation thoroughly before committing real capital.
