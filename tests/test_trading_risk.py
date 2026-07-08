from __future__ import annotations

import unittest

import weatherbet
from trading_risk import (
    contract_matches_strategy,
    extract_market_date,
    fee_adjusted_ev,
    fee_adjusted_kelly,
    market_fee_rate,
)


def valid_market() -> dict:
    return {
        "question": "Will the highest temperature in New York City be 80°F on July 9?",
        "slug": "highest-temperature-in-new-york-city-on-july-9-2026-80f",
        "description": (
            "This market will resolve to the temperature range that contains the "
            "highest temperature recorded by NOAA at LaGuardia Airport in degrees "
            "Fahrenheit on 9 Jul '26. The resolution source is the highest reading "
            "under the Temp column: https://www.weather.gov/wrh/timeseries?site=KLGA"
        ),
        "feesEnabled": True,
        "feeSchedule": {"rate": 0.05, "takerOnly": True},
        "acceptingOrders": True,
        "enableOrderBook": True,
    }


class MarketDateTests(unittest.TestCase):
    def test_extracts_named_month_date_from_real_gamma_slug(self) -> None:
        self.assertEqual(
            extract_market_date(
                "highest-temperature-in-tel-aviv-on-april-7-2026-21c"
            ),
            "2026-04-07",
        )


class ContractValidationTests(unittest.TestCase):
    def test_accepts_exact_high_temperature_station_unit_and_date_contract(self) -> None:
        result = contract_matches_strategy(
            valid_market(),
            city_name="New York City",
            station="KLGA",
            unit="F",
            date_str="2026-07-09",
        )

        self.assertTrue(result.valid, result.reason)

    def test_rejects_non_high_temperature_contract(self) -> None:
        market = valid_market()
        market["question"] = market["question"].replace(
            "highest temperature", "lowest temperature"
        )

        result = contract_matches_strategy(
            market,
            city_name="New York City",
            station="KLGA",
            unit="F",
            date_str="2026-07-09",
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "not_high_temperature")

    def test_rejects_wrong_resolution_station(self) -> None:
        market = valid_market()
        market["description"] = market["description"].replace("KLGA", "KJFK")

        result = contract_matches_strategy(
            market,
            city_name="New York City",
            station="KLGA",
            unit="F",
            date_str="2026-07-09",
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "station_mismatch")

    def test_rejects_wrong_temperature_unit(self) -> None:
        market = valid_market()
        market["question"] = market["question"].replace("°F", "°C")

        result = contract_matches_strategy(
            market,
            city_name="New York City",
            station="KLGA",
            unit="F",
            date_str="2026-07-09",
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "unit_mismatch")

    def test_rejects_resolution_rules_with_wrong_unit(self) -> None:
        market = valid_market()
        market["description"] = market["description"].replace(
            "degrees Fahrenheit", "degrees Celsius"
        )

        result = contract_matches_strategy(
            market,
            city_name="New York City",
            station="KLGA",
            unit="F",
            date_str="2026-07-09",
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "resolution_unit_mismatch")

    def test_rejects_resolution_rules_with_wrong_date(self) -> None:
        market = valid_market()
        market["description"] = market["description"].replace(
            "9 Jul '26", "10 Jul '26"
        )

        result = contract_matches_strategy(
            market,
            city_name="New York City",
            station="KLGA",
            unit="F",
            date_str="2026-07-09",
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "resolution_date_mismatch")

    def test_rejects_wrong_market_date(self) -> None:
        result = contract_matches_strategy(
            valid_market(),
            city_name="New York City",
            station="KLGA",
            unit="F",
            date_str="2026-07-10",
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "date_mismatch")

    def test_rejects_contract_without_open_orderbook(self) -> None:
        market = valid_market()
        market["acceptingOrders"] = False

        result = contract_matches_strategy(
            market,
            city_name="New York City",
            station="KLGA",
            unit="F",
            date_str="2026-07-09",
        )

        self.assertFalse(result.valid)
        self.assertEqual(result.reason, "orders_not_accepted")


class FeeMathTests(unittest.TestCase):
    def test_reads_weather_fee_rate_from_market_schedule(self) -> None:
        self.assertEqual(market_fee_rate(valid_market()), 0.05)

    def test_enabled_market_without_schedule_uses_conservative_weather_rate(self) -> None:
        market = valid_market()
        del market["feeSchedule"]

        self.assertEqual(market_fee_rate(market), 0.05)

    def test_fee_adjusted_ev_includes_entry_taker_fee(self) -> None:
        self.assertAlmostEqual(
            fee_adjusted_ev(probability=0.30, price=0.25, fee_rate=0.05),
            0.1566,
            places=4,
        )

    def test_fee_can_turn_nominal_edge_negative(self) -> None:
        self.assertLess(
            fee_adjusted_ev(probability=0.255, price=0.25, fee_rate=0.05),
            0,
        )

    def test_fee_adjusted_kelly_uses_total_entry_cost(self) -> None:
        self.assertAlmostEqual(
            fee_adjusted_kelly(probability=0.30, price=0.25, fee_rate=0.05),
            0.0549,
            places=4,
        )

    def test_weather_strategy_public_math_accepts_fee_rate(self) -> None:
        self.assertEqual(weatherbet.calc_ev(0.30, 0.25, 0.05), 0.1566)
        self.assertEqual(weatherbet.calc_kelly(0.30, 0.25, 0.05), 0.0137)


if __name__ == "__main__":
    unittest.main()
