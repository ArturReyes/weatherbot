# 🌤 WeatherBet — Polymarket Weather Trading Bot

Automated weather market trading bot for Polymarket. Finds mispriced temperature outcomes using real forecast data from multiple sources across 20 cities worldwide.

No SDK. No black box. Pure Python.

---

## Versions

### `bot_v1.py` — Base Bot
The foundation. Scans 6 US cities, fetches forecasts from NWS using airport station coordinates, finds matching temperature buckets on Polymarket, and enters trades when the market price is below the entry threshold.

No math, no complexity. Just the core logic — good for understanding how the system works.

### `weatherbet.py` — Full Bot (current)
Everything in v1, plus:
- **20 cities** across 4 continents (US, Europe, Asia, South America, Oceania)
- **3 forecast sources** — ECMWF (global), HRRR/GFS (US, hourly), METAR (real-time observations)
- **Expected Value** — skips trades where the math doesn't work
- **Kelly Criterion** — sizes positions based on edge strength
- **Stop-loss + trailing stop** — 20% stop, moves to breakeven at +20%
- **Slippage filter** — skips markets with spread > $0.03
- **Self-calibration** — learns forecast accuracy per city over time
- **Full data storage** — every forecast snapshot, trade, and resolution saved to JSON

### `live_executor.py` — Live Trading
Connects `weatherbet.py`'s strategy engine to the Polymarket CLOB via the official SDK. Full live trading loop with:
- `SecureClient.buy()` / `sell()` order placement
- On-chain USDC balance checks
- Stop-loss, take-profit, trailing stop exits
- Circuit breaker (stops after 5 consecutive errors)
- Telegram notifications
- Atomic state file writes (no corruption on crash)

---

## How It Works

Polymarket runs markets like "Will the highest temperature in Chicago be between 46–47°F on March 7?" These markets are often mispriced — the forecast says 78% likely but the market is trading at 8 cents.

The bot:
1. Fetches forecasts from ECMWF and HRRR via Open-Meteo (free, no key required)
2. Gets real-time observations from METAR airport stations
3. Finds the matching temperature bucket on Polymarket
4. Calculates Expected Value — only enters if the math is positive
5. Sizes the position using fractional Kelly Criterion
6. Monitors stops every 10 minutes, full scan every hour
7. Auto-resolves markets by querying Polymarket API directly

---

## Why Airport Coordinates Matter

Most bots use city center coordinates. That's wrong.

Every Polymarket weather market resolves on a specific airport station. NYC resolves on LaGuardia (KLGA), Dallas on Love Field (KDAL) — not DFW. The difference between city center and airport can be 3–8°F. On markets with 1–2°F buckets, that's the difference between the right trade and a guaranteed loss.

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
cp .env.example .env
```

> `.env` is in `.gitignore` — your secrets stay local. Never commit `.env`.

---

## Usage

### Paper trade (strategy simulation):

```bash
python weatherbet.py           # start the bot — scans every hour
python weatherbet.py status    # balance and open positions
python weatherbet.py report    # full breakdown of all resolved markets
```

### Live trade:

```bash
python live_executor.py                # run the live trading loop
python live_executor.py status         # show balance + positions
python live_executor.py cancel         # cancel all open CLOB orders
python live_executor.py scan           # run one scan cycle, then exit
```

> Paper-trade first to let calibration data accumulate in `data/` — `live_executor.py` reuses it for sigma calibration.

---

## What API Keys Do You Need?

| Key | Required? | Where to get it |
|-----|-----------|-----------------|
| `PK` | **Yes** | Your MetaMask / wallet's Polygon private key (see below) |
| `VC_KEY` | No | https://www.visualcrossing.com/weather-api (free) |
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

### 2. `VC_KEY` — Visual Crossing (for calibration)

Only needed if you want the bot to fetch actual historical temperatures after markets resolve (for per-city sigma calibration).

```bash
# 1. Sign up: https://www.visualcrossing.com/weather-api
# 2. Free tier: 1,000 queries/day
# 3. Dashboard → API Keys → copy
```

> Without this, the bot uses default sigma values (2°F / 1.2°C). It still works — calibration just won't adapt to each city's forecast accuracy.

### 3. Telegram (for push notifications)

```bash
# Talk to @BotFather: /newbot → name it → copy the bot token
# Send any message to your new bot
# Visit: https://api.telegram.org/bot<TOKEN>/getUpdates
# Copy the "chat": { "id": 123456789 }
```

---

## Configure `.env`

```env
# === REQUIRED ===
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
| `max_slippage` | 0.03 | Max allowed ask-bid spread |
| `kelly_fraction` | 0.25 | Fraction of Kelly to bet (0.25 = quarter-Kelly) |
| `scan_interval` | 3600 | Seconds between full market scans |

---

## Pre-Flight Checklist

```bash
# 1. Install everything
pip install -r requirements.txt
pip install --pre polymarket-client

# 2. Set up secrets
cp .env.example .env
# ... edit .env with PK and optional keys ...

# 3. Verify imports
python -c "from weatherbet import bucket_prob; print('weatherbet OK')"
python -c "from live_executor import LiveExecutor; print('live_executor OK')"

# 4. Paper trade for a few days (collect calibration data)
python weatherbet.py status

# 5. Dry-run live (scans + prints signals, attempts orders)
python live_executor.py scan

# 6. Go live
python live_executor.py
```

---

## Data Storage

All data is saved to `data/markets/` — one JSON file per market. Each file contains:
- Hourly forecast snapshots (ECMWF, HRRR, METAR)
- Market price history
- Position details (entry, stop, PnL)
- Final resolution outcome

This data is used for self-calibration — the bot learns forecast accuracy per city over time and adjusts position sizing accordingly.

---

## APIs Used

| API | Auth | Purpose |
|-----|------|---------|
| Open-Meteo | None | ECMWF + HRRR forecasts |
| Aviation Weather (METAR) | None | Real-time station observations |
| Polymarket Gamma | None | Market data |
| Polymarket CLOB | Wallet signature | Order placement (live trading only) |
| Visual Crossing | Free key | Historical temps for calibration |

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
