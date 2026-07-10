from __future__ import annotations

import copy
import dataclasses
import unittest
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import weatherbet
from live_executor import (
    ForecastCache,
    LiveExecutor,
    TradeSignal,
    _extract_date_from_slug,
    calc_kelly,
    generate_signals,
    refresh_signal_with_live_market,
)
from live_trading import OrderSubmission, PositionSnapshot


def make_signal() -> TradeSignal:
    return TradeSignal(
        action="BUY",
        token_id="token-1",
        market_id="market-1",
        condition_id="condition-1",
        city_slug="nyc",
        city_name="New York City",
        date_str="2026-07-09",
        forecast_temp=80,
        bucket_low=80,
        bucket_high=80,
        unit="F",
        probability=0.40,
        entry_price=0.25,
        spread=0.01,
        ev=0.20,
        kelly=0.10,
        size_usdc=2.0,
        shares=8.0,
        forecast_source="ecmwf",
        sigma=2.0,
        raw_forecast_temp=82,
        corrected_forecast_temp=80,
        forecast_bias=2.0,
        forecast_raw_bias=2.0,
        forecast_lead_bucket="24_48h",
        forecast_calibration_n=12,
        raw_probability=0.35,
        raw_ev=0.10,
    )


def make_state() -> dict:
    return {
        "balance_ref": 25.0,
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "positions": [],
        "failed_signals": [],
    }


def fresh_market_detail(*, best_bid: float = 0.24, best_ask: float = 0.25) -> dict:
    return {
        "id": "market-1",
        "question": "Will the highest temperature in New York City be 80°F on July 9?",
        "slug": "highest-temperature-in-new-york-city-on-july-9-2026-80f",
        "description": (
            "The highest temperature recorded by NOAA at LaGuardia Airport in "
            "degrees Fahrenheit on 9 Jul '26. The resolution source is the highest reading "
            "under Temp: https://www.weather.gov/wrh/timeseries?site=KLGA"
        ),
        "endDate": "2026-07-09T23:00:00Z",
        "volume": 1000,
        "acceptingOrders": True,
        "enableOrderBook": True,
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.25", "0.75"]',
        "clobTokenIds": '["token-1", "token-no"]',
        "bestBid": best_bid,
        "bestAsk": best_ask,
        "conditionId": "condition-1",
    }


class FakeGateway:
    def __init__(self) -> None:
        self.buy_result = OrderSubmission(True, "order-1", "matched", None)
        self.sell_result = OrderSubmission(True, "order-2", "matched", None)
        self.positions: dict[str, PositionSnapshot] = {}
        self.buy_error: Exception | None = None
        self.buy_calls: list[tuple[str, Decimal, Decimal]] = []
        self.sell_calls: list[tuple[str, Decimal, Decimal]] = []
        self.sell_price = Decimal("0.18")
        self.sell_price_calls: list[tuple[str, Decimal]] = []
        self.redeem_calls: list[str] = []

    def buy(self, *, token_id: str, amount: Decimal, max_price: Decimal):
        self.buy_calls.append((token_id, amount, max_price))
        if self.buy_error:
            raise self.buy_error
        return self.buy_result

    def sell(self, *, token_id: str, shares: Decimal, min_price: Decimal):
        self.sell_calls.append((token_id, shares, min_price))
        return self.sell_result

    def get_positions(self):
        return self.positions

    def get_executable_sell_price(self, *, token_id: str, shares: Decimal):
        self.sell_price_calls.append((token_id, shares))
        return self.sell_price

    def redeem(self, *, condition_id: str):
        self.redeem_calls.append(condition_id)
        return "0xredeem"

    def get_balance(self):
        return Decimal("25")


class ForecastCacheTests(unittest.TestCase):
    def test_source_ttls_expire_independently(self) -> None:
        now = [1000.0]
        calls: list[str] = []

        def ecmwf(city_slug: str, dates: set[str]) -> dict[str, float]:
            calls.append("ecmwf")
            return {"2026-07-09": 80}

        def hrrr(city_slug: str, dates: set[str]) -> dict[str, float]:
            calls.append("hrrr")
            return {"2026-07-09": 81}

        cache = ForecastCache(
            fetch_ecmwf=ecmwf,
            fetch_hrrr=hrrr,
            fetch_metar=lambda city_slug: None,
            ttl_seconds={"ecmwf": 100, "hrrr": 10, "metar": 5},
            now_fn=lambda: now[0],
        )

        cache.sources_for(
            city_slug="nyc",
            date_str="2026-07-09",
            is_us_city=True,
            hours_to_resolution_value=48,
        )
        now[0] += 11
        cache.sources_for(
            city_slug="nyc",
            date_str="2026-07-09",
            is_us_city=True,
            hours_to_resolution_value=48,
        )

        self.assertEqual(calls.count("ecmwf"), 1)
        self.assertEqual(calls.count("hrrr"), 2)

    def test_metar_uses_short_ttl(self) -> None:
        now = [1000.0]
        calls = 0

        def metar(city_slug: str) -> float:
            nonlocal calls
            calls += 1
            return 79

        cache = ForecastCache(
            fetch_ecmwf=lambda city_slug, dates: {},
            fetch_hrrr=lambda city_slug, dates: {},
            fetch_metar=metar,
            ttl_seconds={"ecmwf": 100, "hrrr": 100, "metar": 45},
            now_fn=lambda: now[0],
        )

        cache.sources_for(
            city_slug="nyc",
            date_str="2026-07-09",
            is_us_city=True,
            hours_to_resolution_value=2,
        )
        now[0] += 44
        cache.sources_for(
            city_slug="nyc",
            date_str="2026-07-09",
            is_us_city=True,
            hours_to_resolution_value=2,
        )
        now[0] += 2
        cache.sources_for(
            city_slug="nyc",
            date_str="2026-07-09",
            is_us_city=True,
            hours_to_resolution_value=2,
        )

        self.assertEqual(calls, 2)


class LiveExecutorEntryTests(unittest.TestCase):
    def test_scan_stops_before_market_io_when_promotion_gate_is_not_ready(self) -> None:
        executor = LiveExecutor(
            private_key="unused",
            gateway=FakeGateway(),
            state=make_state(),
            state_saver=lambda state: None,
        )
        readiness = SimpleNamespace(
            decision=SimpleNamespace(ready=False, reasons=("insufficient holdout",)),
        )

        with (
            patch("live_executor.weatherbet.live_readiness_report", return_value=readiness),
            patch("live_executor.fetch_outdoor_markets") as fetch_markets,
        ):
            placed = executor.scan_and_execute()

        self.assertEqual(placed, 0)
        fetch_markets.assert_not_called()

    def test_no_revalidation_uses_the_no_token_order_book(self) -> None:
        signal = make_signal()
        signal = dataclasses.replace(
            signal,
            token_id="token-no",
            outcome_side="NO",
            probability=0.80,
            entry_price=0.20,
            spread=0.02,
        )
        detail = fresh_market_detail()
        with patch("live_executor.fetch_executable_quote") as fetch_quote:
            fetch_quote.return_value = type("Quote", (), {"bid": 0.18, "ask": 0.20})()
            refreshed = refresh_signal_with_live_market(
                signal,
                detail,
                balance_ref=25.0,
            )

        self.assertIsNotNone(refreshed)
        assert refreshed is not None
        self.assertEqual(refreshed.entry_price, 0.20)
        fetch_quote.assert_called_once_with("token-no")
    def test_real_gamma_slug_date_is_supported(self) -> None:
        self.assertEqual(
            _extract_date_from_slug(
                "highest-temperature-in-tel-aviv-on-april-7-2026-21c"
            ),
            "2026-04-07",
        )

    def test_invalid_contract_is_rejected_before_forecast_io(self) -> None:
        market = {
            "id": "market-1",
            "question": "Will the lowest temperature in New York City be 80°F on July 9?",
            "slug": "lowest-temperature-in-new-york-city-2026-07-09-80f",
            "description": "Lowest temperature, site=KLGA",
            "endDate": "2026-07-09T23:00:00Z",
            "volume": 1000,
            "acceptingOrders": True,
            "enableOrderBook": True,
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.10", "0.90"]',
            "clobTokenIds": '["token-1", "token-no"]',
            "bestBid": 0.09,
            "bestAsk": 0.10,
        }

        with (
            patch("live_executor.get_ecmwf", return_value={}) as forecast,
            patch("live_executor.get_hrrr", return_value={}) as hrrr,
            patch("live_executor.get_metar", return_value=None),
        ):
            signals = generate_signals([market], make_state())

        self.assertEqual(signals, [])
        forecast.assert_not_called()
        hrrr.assert_not_called()

    def test_signal_ev_and_kelly_include_market_fee(self) -> None:
        end_dt = datetime.now(timezone.utc) + timedelta(hours=36)
        date_str = end_dt.strftime("%Y-%m-%d")
        month = end_dt.strftime("%B")
        day = end_dt.day
        short_month = end_dt.strftime("%b")
        short_year = str(end_dt.year)[2:]
        market = {
            "id": "market-1",
            "question": f"Will the highest temperature in New York City be 80°F on {month} {day}?",
            "slug": f"highest-temperature-in-new-york-city-on-{month.lower()}-{day}-{end_dt.year}-80f",
            "description": (
                "The highest temperature recorded by NOAA at LaGuardia Airport in "
                f"degrees Fahrenheit on {day} {short_month} '{short_year}. The resolution source is the highest reading "
                "under Temp: https://www.weather.gov/wrh/timeseries?site=KLGA"
            ),
            "endDate": end_dt.isoformat(),
            "volume": 1000,
            "acceptingOrders": True,
            "enableOrderBook": True,
            "feesEnabled": True,
            "feeSchedule": {"rate": 0.01},
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.10", "0.90"]',
            "clobTokenIds": '["token-1", "token-no"]',
            "bestBid": 0.09,
            "bestAsk": 0.10,
            "conditionId": "condition-1",
        }
        detail = dict(market)
        detail["feeSchedule"] = {"rate": 0.05}

        with (
            patch("live_executor.get_ecmwf", return_value={date_str: 80}),
            patch("live_executor.get_hrrr", return_value={}),
            patch("live_executor.get_metar", return_value=None),
            patch("live_executor.fetch_market_detail", return_value=detail),
        ):
            signals = generate_signals([market], make_state())

        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].fee_rate, 0.05)
        self.assertLess(signals[0].ev, 0.974)
        self.assertEqual(
            signals[0].kelly,
            calc_kelly(signals[0].probability, signals[0].entry_price, 0.05),
        )

    def test_signal_generation_uses_bias_corrected_forecast_for_probability(self) -> None:
        end_dt = datetime.now(timezone.utc) + timedelta(hours=36)
        date_str = end_dt.strftime("%Y-%m-%d")
        month = end_dt.strftime("%B").lower()
        day = end_dt.day
        year = end_dt.year
        short_month = end_dt.strftime("%b")
        short_year = str(year)[2:]
        market = {
            "id": "market-1",
            "question": f"Will the highest temperature in New York City be 80°F on {month.title()} {day}?",
            "slug": f"highest-temperature-in-new-york-city-on-{month}-{day}-{year}-80f",
            "description": (
                "The highest temperature recorded by NOAA at LaGuardia Airport in "
                f"degrees Fahrenheit on {day} {short_month} '{short_year}. "
                "The resolution source is the highest reading under Temp: "
                "https://www.weather.gov/wrh/timeseries?site=KLGA"
            ),
            "endDate": end_dt.isoformat().replace("+00:00", "Z"),
            "volume": 1000,
            "acceptingOrders": True,
            "enableOrderBook": True,
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.10", "0.90"]',
            "clobTokenIds": '["token-1", "token-no"]',
            "bestBid": 0.09,
            "bestAsk": 0.10,
            "conditionId": "condition-1",
        }
        original_cal = weatherbet._cal
        try:
            weatherbet.install_calibration({
                "nyc_ecmwf_24_48h": {
                    "bias": 2.0,
                    "raw_bias": 2.2,
                    "sigma": 2.0,
                    "n": 12,
                }
            })
            with (
                patch("live_executor.get_ecmwf", return_value={date_str: 82}),
                patch("live_executor.get_hrrr", return_value={}),
                patch("live_executor.fetch_market_detail", return_value=market),
            ):
                signals = generate_signals([market], make_state())
        finally:
            weatherbet.install_calibration(original_cal)

        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertEqual(signal.raw_forecast_temp, 82.0)
        self.assertEqual(signal.corrected_forecast_temp, 80.0)
        self.assertEqual(signal.forecast_temp, 80.0)
        self.assertEqual(signal.forecast_bias, 2.0)
        self.assertEqual(signal.forecast_raw_bias, 2.2)
        self.assertEqual(signal.forecast_lead_bucket, "24_48h")
        self.assertEqual(signal.forecast_calibration_n, 12)
        self.assertLess(signal.raw_probability, signal.probability)
        self.assertLess(signal.raw_ev, signal.ev)

    def test_signal_generation_reuses_forecast_cache_for_same_city_date(self) -> None:
        market_1 = {
            "id": "market-1",
            "question": "Will the highest temperature in New York City be 80°F on July 10?",
            "slug": "highest-temperature-in-new-york-city-on-july-10-2026-80f",
            "description": (
                "The highest temperature recorded by NOAA at LaGuardia Airport in "
                "degrees Fahrenheit on 10 Jul '26. The resolution source is the highest reading "
                "under Temp: https://www.weather.gov/wrh/timeseries?site=KLGA"
            ),
            "endDate": "2026-07-10T23:00:00Z",
            "volume": 1000,
            "acceptingOrders": True,
            "enableOrderBook": True,
            "outcomes": '["Yes", "No"]',
            "outcomePrices": '["0.10", "0.90"]',
            "clobTokenIds": '["token-1", "token-no-1"]',
            "bestBid": 0.09,
            "bestAsk": 0.10,
            "conditionId": "condition-1",
        }
        market_2 = dict(market_1)
        market_2["id"] = "market-2"
        market_2["conditionId"] = "condition-2"
        market_2["clobTokenIds"] = '["token-2", "token-no-2"]'

        details = {"market-1": market_1, "market-2": market_2}

        with (
            patch("live_executor.get_ecmwf", return_value={"2026-07-10": 80}) as ecmwf,
            patch("live_executor.get_hrrr", return_value={}),
            patch("live_executor.fetch_market_detail", side_effect=lambda mid: details[mid]),
        ):
            signals = generate_signals([market_1, market_2], make_state())

        self.assertEqual(len(signals), 2)
        ecmwf.assert_called_once_with("nyc", {"2026-07-10"})

    def test_entry_persists_intent_before_submission_and_uses_exchange_fill(self) -> None:
        gateway = FakeGateway()
        gateway.positions = {
            "token-1": PositionSnapshot(
                "token-1", "condition-1", Decimal("4"), Decimal("0.25")
            )
        }
        saved: list[dict] = []
        executor = LiveExecutor(
            private_key="unused",
            gateway=gateway,
            state=make_state(),
            state_saver=lambda state: saved.append(copy.deepcopy(state)),
        )

        with patch("live_executor.fetch_market_detail", return_value=fresh_market_detail()):
            placed = executor._execute_signal(make_signal())

        self.assertTrue(placed)
        self.assertEqual(saved[0]["positions"][0]["status"], "submitting")
        position = executor._state["positions"][0]
        self.assertEqual(position["status"], "open")
        self.assertEqual(position["shares"], 4.0)
        self.assertEqual(position["amount"], 1.0)
        self.assertEqual(position["order_id"], "order-1")
        self.assertEqual(position["raw_forecast_temp"], 82)
        self.assertEqual(position["corrected_forecast_temp"], 80)
        self.assertEqual(position["forecast_bias"], 2.0)
        self.assertEqual(position["forecast_lead_bucket"], "24_48h")
        self.assertEqual(position["forecast_calibration_n"], 12)
        self.assertEqual(position["raw_probability"], 0.35)
        self.assertEqual(position["raw_ev"], 0.4)

    def test_risk_gate_rejects_event_overexposure_before_submission(self) -> None:
        gateway = FakeGateway()
        state = make_state()
        state["positions"].append({
            "status": "open",
            "amount": 1.0,
            "city_slug": "nyc",
            "date": "2026-07-09",
        })
        executor = LiveExecutor(
            private_key="unused",
            gateway=gateway,
            state=state,
            state_saver=lambda state: None,
        )

        with patch("live_executor.fetch_market_detail") as fetch_detail:
            placed = executor._execute_signal(make_signal())

        self.assertFalse(placed)
        self.assertEqual(len(executor._state["positions"]), 1)
        fetch_detail.assert_not_called()

    def test_ambiguous_submission_error_keeps_blocking_unknown_intent(self) -> None:
        gateway = FakeGateway()
        gateway.buy_error = ConnectionError("response lost")
        executor = LiveExecutor(
            private_key="unused",
            gateway=gateway,
            state=make_state(),
            state_saver=lambda state: None,
        )

        with patch("live_executor.fetch_market_detail", return_value=fresh_market_detail()):
            placed = executor._execute_signal(make_signal())

        self.assertFalse(placed)
        self.assertEqual(executor._state["positions"][0]["status"], "unknown")

    def test_rejected_submission_does_not_create_an_active_position(self) -> None:
        gateway = FakeGateway()
        gateway.buy_result = OrderSubmission(
            False, None, "rejected", "fak_not_filled: no liquidity"
        )
        executor = LiveExecutor(
            private_key="unused",
            gateway=gateway,
            state=make_state(),
            state_saver=lambda state: None,
        )

        with patch("live_executor.fetch_market_detail", return_value=fresh_market_detail()):
            placed = executor._execute_signal(make_signal())

        self.assertFalse(placed)
        self.assertEqual(executor._state["positions"][0]["status"], "rejected")
        self.assertEqual(executor._state["total_trades"], 0)

    def test_execution_uses_fresh_market_ask_before_submission(self) -> None:
        gateway = FakeGateway()
        gateway.positions = {
            "token-1": PositionSnapshot(
                "token-1", "condition-1", Decimal("4"), Decimal("0.30")
            )
        }
        executor = LiveExecutor(
            private_key="unused",
            gateway=gateway,
            state=make_state(),
            state_saver=lambda state: None,
        )

        with patch(
            "live_executor.fetch_market_detail",
            return_value=fresh_market_detail(best_bid=0.29, best_ask=0.30),
        ):
            placed = executor._execute_signal(make_signal())

        self.assertTrue(placed)
        self.assertEqual(gateway.buy_calls[0][2], Decimal("0.3"))
        self.assertEqual(executor._state["positions"][0]["entry_price"], 0.3)

    def test_execution_rejects_when_fresh_market_ask_is_too_high(self) -> None:
        gateway = FakeGateway()
        executor = LiveExecutor(
            private_key="unused",
            gateway=gateway,
            state=make_state(),
            state_saver=lambda state: None,
        )

        with patch(
            "live_executor.fetch_market_detail",
            return_value=fresh_market_detail(best_bid=0.44, best_ask=0.46),
        ):
            placed = executor._execute_signal(make_signal())

        self.assertFalse(placed)
        self.assertEqual(gateway.buy_calls, [])
        self.assertEqual(executor._state["positions"], [])


class LiveExecutorExitTests(unittest.TestCase):
    def test_position_monitor_uses_depth_aware_clob_sell_price(self) -> None:
        gateway = FakeGateway()
        gateway.sell_price = Decimal("0.19")
        gateway.positions = {
            "token-1": PositionSnapshot(
                "token-1", "condition-1", Decimal("4"), Decimal("0.25")
            )
        }
        state = make_state()
        state["positions"].append(
            {
                "market_id": "market-1",
                "token_id": "token-1",
                "condition_id": "condition-1",
                "status": "open",
                "shares": 4.0,
                "amount": 1.0,
                "entry_price": 0.25,
                "stop_price": 0.20,
                "city_name": "New York City",
                "date": "2026-07-09",
                "bucket_low": 80,
                "bucket_high": 80,
                "unit": "F",
            }
        )
        executor = LiveExecutor(
            private_key="unused",
            gateway=gateway,
            state=state,
            state_saver=lambda state: None,
        )

        with patch(
            "live_executor.fetch_market_detail",
            return_value={
                "endDate": "2026-07-09T23:00:00Z",
                "outcomePrices": '["0.50", "0.50"]',
                "clobTokenIds": '["token-1", "token-no"]',
            },
        ):
            executor._check_positions()

        self.assertEqual(
            gateway.sell_price_calls,
            [("token-1", Decimal("4.0"))],
        )
        self.assertEqual(len(gateway.sell_calls), 1)

    def test_redeemable_position_submits_and_confirms_redemption(self) -> None:
        gateway = FakeGateway()
        gateway.positions = {
            "token-1": PositionSnapshot(
                "token-1",
                "condition-1",
                Decimal("4"),
                Decimal("0.25"),
                redeemable=True,
                current_price=Decimal("1"),
            )
        }
        state = make_state()
        position = {
            "market_id": "market-1",
            "token_id": "token-1",
            "condition_id": "condition-1",
            "status": "open",
            "shares": 4.0,
            "amount": 1.0,
            "entry_price": 0.25,
            "city_name": "New York City",
            "date": "2026-07-09",
            "bucket_low": 80,
            "bucket_high": 80,
            "unit": "F",
        }
        state["positions"].append(position)
        executor = LiveExecutor(
            private_key="unused",
            gateway=gateway,
            state=state,
            state_saver=lambda state: None,
        )

        executor._check_positions()

        self.assertEqual(gateway.redeem_calls, ["condition-1"])
        self.assertEqual(position["status"], "redemption_confirmed")
        self.assertEqual(position["redemption_tx"], "0xredeem")
        self.assertEqual(position["pnl"], 3.0)
    def test_exit_sells_exchange_quantity_and_keeps_partial_residual_open(self) -> None:
        gateway = FakeGateway()
        gateway.positions = {
            "token-1": PositionSnapshot(
                "token-1", "condition-1", Decimal("2"), Decimal("0.25")
            )
        }
        state = make_state()
        position = {
            "token_id": "token-1",
            "condition_id": "condition-1",
            "status": "open",
            "shares": 4.0,
            "entry_price": 0.25,
            "city_name": "New York City",
            "date": "2026-07-09",
            "bucket_low": 80,
            "bucket_high": 80,
            "unit": "F",
        }
        state["positions"].append(position)
        executor = LiveExecutor(
            private_key="unused",
            gateway=gateway,
            state=state,
            state_saver=lambda state: None,
        )

        closed = executor._exit_position(position, 0.40, "take_profit")

        self.assertFalse(closed)
        self.assertEqual(
            gateway.sell_calls,
            [("token-1", Decimal("4.0"), Decimal("0.380"))],
        )
        self.assertEqual(position["status"], "open")
        self.assertEqual(position["shares"], 2.0)

    def test_exit_with_stale_exchange_snapshot_remains_pending(self) -> None:
        gateway = FakeGateway()
        gateway.positions = {
            "token-1": PositionSnapshot(
                "token-1", "condition-1", Decimal("4"), Decimal("0.25")
            )
        }
        state = make_state()
        position = {
            "token_id": "token-1",
            "condition_id": "condition-1",
            "status": "open",
            "shares": 4.0,
            "entry_price": 0.25,
            "city_name": "New York City",
            "date": "2026-07-09",
            "bucket_low": 80,
            "bucket_high": 80,
            "unit": "F",
        }
        state["positions"].append(position)
        executor = LiveExecutor(
            private_key="unused",
            gateway=gateway,
            state=state,
            state_saver=lambda state: None,
        )

        closed = executor._exit_position(position, 0.40, "take_profit")

        self.assertFalse(closed)
        self.assertEqual(position["status"], "exit_pending")

    def test_unmanaged_exchange_position_opens_circuit_breaker(self) -> None:
        gateway = FakeGateway()
        gateway.positions = {
            "token-x": PositionSnapshot(
                "token-x", "condition-x", Decimal("2"), Decimal("0.25")
            )
        }
        executor = LiveExecutor(
            private_key="unused",
            gateway=gateway,
            state=make_state(),
            state_saver=lambda state: None,
        )

        report = executor.reconcile_exchange_state()

        self.assertTrue(report.has_discrepancies)
        self.assertTrue(executor._circuit_open)
        self.assertEqual(executor._state["positions"][0]["status"], "unmanaged")

    def test_status_report_handles_unmanaged_position_metadata(self) -> None:
        gateway = FakeGateway()
        state = make_state()
        state["positions"].append(
            {
                "token_id": "token-x",
                "condition_id": "condition-x",
                "status": "unmanaged",
                "shares": 2.0,
                "entry_price": 0.25,
            }
        )
        executor = LiveExecutor(
            private_key="unused",
            gateway=gateway,
            state=state,
            state_saver=lambda state: None,
        )

        report = executor.status_report()

        self.assertIn("UNMANAGED", report)
        self.assertIn("token-x", report)


if __name__ == "__main__":
    unittest.main()
