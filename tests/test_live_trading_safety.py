import importlib
import os
import sys
import types
import unittest
from unittest.mock import patch


class _DummyApiCreds:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyOrderArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyMarketOrderArgs:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _DummyOrderType:
    FOK = "FOK"
    GTC = "GTC"


class _DummyBalanceAllowanceParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

class _DummyAssetType:
    CONDITIONAL = "CONDITIONAL"

def _install_dependency_stubs():
    requests_module = types.ModuleType("requests")
    requests_module.get = lambda *args, **kwargs: None

    py_clob_client_module = types.ModuleType("py_clob_client_v2")
    client_module = types.ModuleType("py_clob_client_v2.client")
    clob_types_module = types.ModuleType("py_clob_client_v2.clob_types")

    client_module.ClobClient = object
    clob_types_module.ApiCreds = _DummyApiCreds
    clob_types_module.MarketOrderArgs = _DummyMarketOrderArgs
    clob_types_module.OrderArgs = _DummyOrderArgs
    clob_types_module.OrderType = _DummyOrderType
    clob_types_module.BalanceAllowanceParams = _DummyBalanceAllowanceParams
    clob_types_module.AssetType = _DummyAssetType

    sys.modules["requests"] = requests_module
    sys.modules["py_clob_client_v2"] = py_clob_client_module
    sys.modules["py_clob_client_v2.client"] = client_module
    sys.modules["py_clob_client_v2.clob_types"] = clob_types_module


def _load_bot(env):
    _install_dependency_stubs()
    for name in (
        "PRIVATE_KEY",
        "CHAIN_ID",
        "SIGNATURE_TYPE",
        "PROXY_KEY",
        "WALLET_ADDRESS",
        "POLY_API_KEY",
        "POLY_SECRET",
        "POLY_PASSPHRASE",
    ):
        os.environ.pop(name, None)
    os.environ.update(env)
    sys.modules.pop("bot_v2", None)
    return importlib.import_module("bot_v2")


class LiveTradingSafetyTests(unittest.TestCase):
    def test_live_trading_requires_all_clob_credentials(self):
        bot = _load_bot({"PRIVATE_KEY": "set", "POLY_API_KEY": "set"})

        self.assertFalse(bot.LIVE_TRADING)
        self.assertIn("POLY_SECRET", bot.LIVE_CONFIG_ERRORS)
        self.assertIn("POLY_PASSPHRASE", bot.LIVE_CONFIG_ERRORS)
        self.assertIn("PROXY_KEY or WALLET_ADDRESS", bot.LIVE_CONFIG_ERRORS)

    def test_order_response_success_requires_explicit_success_and_order_id(self):
        bot = _load_bot({})

        self.assertFalse(bot.live_order_succeeded({"success": False, "orderID": "abc"}))
        self.assertFalse(bot.live_order_succeeded({"success": True}))
        self.assertFalse(bot.live_order_succeeded({"success": True, "errorMsg": "rejected", "orderID": "abc"}))
        self.assertTrue(bot.live_order_succeeded({"success": True, "orderID": "abc"}))

    def test_live_buy_rejects_unsuccessful_clob_response(self):
        bot = _load_bot({
            "PRIVATE_KEY": "set",
            "POLY_API_KEY": "set",
            "POLY_SECRET": "set",
            "POLY_PASSPHRASE": "set",
            "PROXY_KEY": "0xproxy",
        })

        class FakeClient:
            def create_market_order(self, args):
                return {"signed": True}

            def post_order(self, signed, order_type):
                return {"success": False, "orderID": "abc"}

        with patch.object(bot, "get_clob", return_value=FakeClient()):
            self.assertFalse(bot.place_live_buy("token", 0.25, 1.00))

    def test_live_buy_uses_installed_clob_create_order_api(self):
        bot = _load_bot({
            "PRIVATE_KEY": "set",
            "POLY_API_KEY": "set",
            "POLY_SECRET": "set",
            "POLY_PASSPHRASE": "set",
            "PROXY_KEY": "0xproxy",
        })

        class FakeClient:
            def __init__(self):
                self.created = False

            def create_market_order(self, args):
                self.created = True
                return {"signed": True}

            def post_order(self, signed, order_type):
                return {"success": True, "orderID": "abc"}

        fake = FakeClient()

        with patch.object(bot, "get_clob", return_value=fake):
            self.assertTrue(bot.place_live_buy("token", 0.25, 1.00))
        self.assertTrue(fake.created)

    def test_live_buy_uses_market_order_amount_with_cent_precision(self):
        bot = _load_bot({
            "PRIVATE_KEY": "set",
            "POLY_API_KEY": "set",
            "POLY_SECRET": "set",
            "POLY_PASSPHRASE": "set",
            "PROXY_KEY": "0xproxy",
        })

        class FakeClient:
            def __init__(self):
                self.order_args = None

            def create_market_order(self, args):
                self.order_args = args
                return {"signed": True}

            def post_order(self, signed, order_type):
                return {"success": True, "orderID": "abc"}

        fake = FakeClient()

        with patch.object(bot, "get_clob", return_value=fake):
            self.assertTrue(bot.place_live_buy("token", 0.3333333, 1.999999))

        self.assertEqual(fake.order_args.kwargs["amount"], 2.00)
        self.assertEqual(fake.order_args.kwargs["price"], 0.3333)
        self.assertEqual(fake.order_args.kwargs["order_type"], bot.OrderType.FOK)

    def test_live_buy_passes_wallet_balance_for_fee_adjusted_market_buy(self):
        bot = _load_bot({
            "PRIVATE_KEY": "set",
            "POLY_API_KEY": "set",
            "POLY_SECRET": "set",
            "POLY_PASSPHRASE": "set",
            "WALLET_ADDRESS": "0x1111111111111111111111111111111111111111",
        })

        class FakeClient:
            def __init__(self):
                self.order_args = None

            def create_market_order(self, args):
                self.order_args = args
                return {"signed": True}

            def post_order(self, signed, order_type):
                return {"success": True, "orderID": "abc"}

        fake = FakeClient()

        with patch.object(bot, "get_clob", return_value=fake), patch.object(bot, "get_wallet_pusd_balance", return_value=20.527167):
            self.assertTrue(bot.place_live_buy("token", 0.25, 1.00))

        self.assertEqual(fake.order_args.kwargs["user_usdc_balance"], 20.527167)

    def test_live_buy_retries_transient_clob_request_exception(self):
        bot = _load_bot({
            "PRIVATE_KEY": "set",
            "POLY_API_KEY": "set",
            "POLY_SECRET": "set",
            "POLY_PASSPHRASE": "set",
            "PROXY_KEY": "0xproxy",
        })

        class RequestException(Exception):
            status_code = None

            def __str__(self):
                return "PolyApiException[status_code=None, error_message=Request exception!]"

        class FakeClient:
            def __init__(self):
                self.posts = 0

            def create_market_order(self, args):
                return {"signed": True}

            def post_order(self, signed, order_type):
                self.posts += 1
                if self.posts == 1:
                    raise RequestException()
                return {"success": True, "orderID": "abc"}

        fake = FakeClient()

        with patch.object(bot, "get_clob", return_value=fake), patch.object(bot.time, "sleep"):
            self.assertTrue(bot.place_live_buy("token", 0.25, 1.00))

        self.assertEqual(fake.posts, 2)

    def test_resolve_retries_gamma_timeout(self):
        bot = _load_bot({})

        class FakeResponse:
            def json(self):
                return {"closed": False}

        calls = {"count": 0}

        def fake_get(url, timeout):
            calls["count"] += 1
            if calls["count"] == 1:
                raise TimeoutError("Read timed out")
            return FakeResponse()

        with patch.object(bot.requests, "get", side_effect=fake_get), patch.object(bot.time, "sleep"):
            self.assertIsNone(bot.check_market_resolved("2057076"))

        self.assertEqual(calls["count"], 2)

    def test_live_buy_treats_fok_not_filled_as_skip(self):
        bot = _load_bot({
            "PRIVATE_KEY": "set",
            "POLY_API_KEY": "set",
            "POLY_SECRET": "set",
            "POLY_PASSPHRASE": "set",
            "PROXY_KEY": "0xproxy",
        })

        class FokNotFilled(Exception):
            status_code = 400
            error_msg = {
                "error": "order couldn't be fully filled. FOK orders are fully filled or killed.",
                "orderID": "0xabc",
            }

        class FakeClient:
            def create_market_order(self, args):
                return {"signed": True}

            def post_order(self, signed, order_type):
                raise FokNotFilled()

        with patch.object(bot, "get_clob", return_value=FakeClient()):
            self.assertFalse(bot.place_live_buy("token", 0.25, 1.00))

    def test_monitor_positions_saves_market_when_trailing_stop_activates(self):
        bot = _load_bot({})
        market = {
            "city": "nyc",
            "date": "2026-04-28",
            "event_end_date": "2026-04-29T00:00:00+00:00",
            "forecast_snapshots": [{"best": 72}],
            "position": {
                "status": "open",
                "market_id": "market-1",
                "entry_price": 0.20,
                "bucket_low": 70,
                "bucket_high": 75,
                "shares": 5,
                "cost": 1.00,
            },
        }

        with (
            patch.object(bot, "load_all_markets", return_value=[market]),
            patch.object(bot, "load_state", return_value={"balance": 10.0}),
            patch.object(bot, "get_gamma_json", return_value={"bestBid": "0.24"}),
            patch.object(bot, "hours_to_resolution", return_value=12),
            patch.object(bot, "save_market") as save_market,
        ):
            self.assertEqual(bot.monitor_positions(), 0)

        self.assertEqual(market["position"]["stop_price"], 0.20)
        self.assertTrue(market["position"]["trailing_activated"])
        save_market.assert_called_once_with(market)

    def test_monitor_positions_uses_outcome_price_when_best_bid_missing(self):
        bot = _load_bot({})
        market = {
            "city": "paris",
            "date": "2026-04-29",
            "event_end_date": "2026-04-29T00:00:00+00:00",
            "forecast_snapshots": [{"best": 20}],
            "all_outcomes": [
                {"market_id": "market-1", "bid": 0.99, "price": 0.99},
            ],
            "position": {
                "status": "open",
                "market_id": "market-1",
                "entry_price": 0.30,
                "stop_price": 0.24,
                "bucket_low": 23,
                "bucket_high": 23,
                "shares": 5,
                "cost": 1.50,
            },
        }

        with (
            patch.object(bot, "load_all_markets", return_value=[market]),
            patch.object(bot, "load_state", return_value={"balance": 10.0}),
            patch.object(bot, "get_gamma_json", return_value={"bestBid": None, "outcomePrices": "[\"0\", \"1\"]"}),
            patch.object(bot, "hours_to_resolution", return_value=1),
            patch.object(bot, "save_market") as save_market,
            patch.object(bot, "save_state") as save_state,
        ):
            self.assertEqual(bot.monitor_positions(), 1)

        pos = market["position"]
        self.assertEqual(pos["status"], "closed")
        self.assertEqual(pos["close_reason"], "stop_loss")
        self.assertEqual(pos["exit_price"], 0.0)
        self.assertEqual(pos["pnl"], -1.5)
        save_market.assert_called_once_with(market)
        save_state.assert_called_once_with({"balance": 10.0})

    def test_monitor_positions_closes_on_wipeout_even_when_forecast_says_in_bucket(self):
        """Wipeout branch: market price ≤ WIPEOUT_PRICE must override forecast_invalid=False
        and close the position. This is the Munich/Tokyo/Seoul/London pattern."""
        bot = _load_bot({})
        market = {
            "city": "seoul",
            "date": "2026-05-03",
            "event_end_date": "2026-05-03T15:00:00+00:00",
            "forecast_snapshots": [{"best": 16.0}],  # forecast says 16 — IN bucket
            "position": {
                "status": "open",
                "market_id": "market-1",
                "entry_price": 0.26,
                "stop_price": 0.24,
                "bucket_low": 16,
                "bucket_high": 16,
                "shares": 5,
                "cost": 1.30,
            },
        }

        with (
            patch.object(bot, "load_all_markets", return_value=[market]),
            patch.object(bot, "load_state", return_value={"balance": 5.0}),
            patch.object(bot, "get_gamma_json", return_value={"bestBid": "0.001"}),
            patch.object(bot, "hours_to_resolution", return_value=2),
            patch.object(bot, "save_market") as save_market,
            patch.object(bot, "save_state"),
        ):
            self.assertEqual(bot.monitor_positions(), 1)

        pos = market["position"]
        self.assertEqual(pos["status"], "closed")
        self.assertEqual(pos["close_reason"], "stop_loss")
        self.assertAlmostEqual(pos["exit_price"], 0.001)
        save_market.assert_called_once_with(market)

    def test_monitor_positions_holds_on_temporary_dip_when_forecast_in_bucket(self):
        """Negative case: price below stop but ABOVE WIPEOUT_PRICE, forecast still in bucket.
        Should NOT close — this protects against false positives from spread/liquidity dips."""
        bot = _load_bot({})
        market = {
            "city": "london",
            "date": "2026-05-04",
            "event_end_date": "2026-05-04T15:00:00+00:00",
            "forecast_snapshots": [{"best": 19.0}],  # forecast IN bucket
            "position": {
                "status": "open",
                "market_id": "market-1",
                "entry_price": 0.30,
                "stop_price": 0.24,
                "bucket_low": 19,
                "bucket_high": 19,
                "shares": 5,
                "cost": 1.50,
            },
        }

        with (
            patch.object(bot, "load_all_markets", return_value=[market]),
            patch.object(bot, "load_state", return_value={"balance": 5.0}),
            # 0.05 is below stop (0.24) but well above WIPEOUT_PRICE (0.01)
            patch.object(bot, "get_gamma_json", return_value={"bestBid": "0.05"}),
            patch.object(bot, "hours_to_resolution", return_value=2),
            patch.object(bot, "save_market") as save_market,
            patch.object(bot, "save_state") as save_state,
        ):
            self.assertEqual(bot.monitor_positions(), 0)

        self.assertEqual(market["position"]["status"], "open")
        save_market.assert_not_called()
        save_state.assert_not_called()


if __name__ == "__main__":
    unittest.main()
