# Product

## Register

product

## Users

WeatherBet is used by a trading operator who needs to monitor paper and live weather-market execution from a terminal. The user is evaluating whether the bot found a legitimate mispriced bucket, whether it opened or skipped a position, and whether the current portfolio state is safe enough to keep running.

## Product Purpose

WeatherBet scans Polymarket temperature markets, compares live prices against calibrated weather forecasts, and paper/live trades only when the strategy clears risk and expected-value gates. Success means the operator can quickly understand what the bot did, why it did it, and whether any action requires attention.

## Brand Personality

Pragmatic, precise, calm. The product should feel like a serious operator console: fast to scan, low-noise, and direct about risk.

## Anti-references

Avoid decorative dashboards, chatty bot narration, noisy log spam, color-only status meaning, and output that hides the decision path behind generic "ok" messages. Do not make the terminal feel like a marketing surface.

## Design Principles

1. Lead with operator state: show what changed, what was skipped, and what needs attention.
2. Keep routine success quiet: cities with no action should be visible but not visually dominant.
3. Preserve the audit trail where it matters: buys, closes, skips, warnings, and resolutions need concrete numbers.
4. Use compact hierarchy: grouped scan phases beat interleaved progress logs.
5. Make risk legible: status labels, symbols, and color should reinforce each other, never replace each other.

## Accessibility & Inclusion

Terminal output may use Unicode and ANSI color, but meaning must also be present in plain text labels. The console should remain understandable when copied into logs, viewed without color, or read under time pressure.
