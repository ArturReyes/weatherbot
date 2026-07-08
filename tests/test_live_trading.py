from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from live_trading import (
    OrderSubmission,
    PolymarketGateway,
    PositionSnapshot,
    ProcessLock,
    ProcessLockError,
    reconcile_entry,
    reconcile_exit,
    reconcile_state,
)


@dataclass
class FakeBalance:
    balance: Decimal


@dataclass
class FakeAcceptedOrder:
    ok: bool = True
    order_id: str = "order-1"
    status: str = "matched"


@dataclass
class FakeRejectedOrder:
    ok: bool = False
    code: str = "fak_not_filled"
    message: str = "no liquidity"


@dataclass
class FakePosition:
    token_id: str
    condition_id: str
    size: Decimal
    avg_price: Decimal
    redeemable: bool = False
    cur_price: Decimal = Decimal("0.5")


@dataclass
class FakeTransactionOutcome:
    transaction_hash: str


class FakeTransactionHandle:
    def __init__(self, outcome: FakeTransactionOutcome) -> None:
        self.outcome = outcome
        self.waited = False

    def wait(self) -> FakeTransactionOutcome:
        self.waited = True
        return self.outcome


class FakePaginator:
    def __init__(self, items: list[object]) -> None:
        self._items = items

    def __iter__(self):
        return iter((object(),))

    def iter_items(self):
        return iter(self._items)


class FakeClient:
    def __init__(self) -> None:
        self.balance_calls: list[dict[str, str]] = []
        self.order_calls: list[dict[str, object]] = []
        self.order_response: object = FakeAcceptedOrder()
        self.positions: list[FakePosition] = []
        self.estimate_calls: list[dict[str, object]] = []
        self.redeem_calls: list[dict[str, str]] = []
        self.transaction_handle = FakeTransactionHandle(
            FakeTransactionOutcome("0xtransaction")
        )

    def get_balance_allowance(self, **kwargs):
        self.balance_calls.append(kwargs)
        return FakeBalance(Decimal("12.34"))

    def place_market_order(self, **kwargs):
        self.order_calls.append(kwargs)
        return self.order_response

    def list_positions(self, **kwargs):
        return FakePaginator(self.positions)

    def estimate_market_price(self, **kwargs):
        self.estimate_calls.append(kwargs)
        return Decimal("0.18")

    def redeem_positions(self, **kwargs):
        self.redeem_calls.append(kwargs)
        return self.transaction_handle


class PolymarketGatewayTests(unittest.TestCase):
    def test_balance_requests_collateral_explicitly(self) -> None:
        client = FakeClient()

        balance = PolymarketGateway(client).get_balance()

        self.assertEqual(balance, Decimal("12.34"))
        self.assertEqual(client.balance_calls, [{"asset_type": "COLLATERAL"}])

    def test_buy_uses_fak_market_order_and_normalizes_acceptance(self) -> None:
        client = FakeClient()

        result = PolymarketGateway(client).buy(
            token_id="token-1",
            amount=Decimal("5.00"),
            max_price=Decimal("0.30"),
        )

        self.assertEqual(
            client.order_calls,
            [{
                "token_id": "token-1",
                "side": "BUY",
                "amount": Decimal("5.00"),
                "max_price": Decimal("0.30"),
                "order_type": "FAK",
            }],
        )
        self.assertEqual(result, OrderSubmission(True, "order-1", "matched", None))

    def test_rejected_order_is_a_result_not_an_exception(self) -> None:
        client = FakeClient()
        client.order_response = FakeRejectedOrder()

        result = PolymarketGateway(client).buy(
            token_id="token-1",
            amount=Decimal("5.00"),
            max_price=Decimal("0.30"),
        )

        self.assertEqual(
            result,
            OrderSubmission(False, None, "rejected", "fak_not_filled: no liquidity"),
        )

    def test_sell_uses_actual_shares_and_fak_market_order(self) -> None:
        client = FakeClient()

        result = PolymarketGateway(client).sell(
            token_id="token-1",
            shares=Decimal("7.5"),
            min_price=Decimal("0.20"),
        )

        self.assertEqual(
            client.order_calls,
            [{
                "token_id": "token-1",
                "side": "SELL",
                "shares": Decimal("7.5"),
                "min_price": Decimal("0.20"),
                "order_type": "FAK",
            }],
        )
        self.assertTrue(result.accepted)

    def test_positions_are_normalized_from_exchange_values(self) -> None:
        client = FakeClient()
        client.positions = [
            FakePosition("token-1", "condition-1", Decimal("7.5"), Decimal("0.22"))
        ]

        positions = PolymarketGateway(client).get_positions()

        self.assertEqual(
            positions,
            {
                "token-1": PositionSnapshot(
                    token_id="token-1",
                    condition_id="condition-1",
                    shares=Decimal("7.5"),
                    average_price=Decimal("0.22"),
                    current_price=Decimal("0.5"),
                )
            },
        )

    def test_redeemable_position_metadata_is_preserved(self) -> None:
        client = FakeClient()
        client.positions = [
            FakePosition(
                "token-1",
                "condition-1",
                Decimal("7.5"),
                Decimal("0.22"),
                redeemable=True,
                cur_price=Decimal("1"),
            )
        ]

        position = PolymarketGateway(client).get_positions()["token-1"]

        self.assertTrue(position.redeemable)
        self.assertEqual(position.current_price, Decimal("1"))

    def test_sell_price_uses_depth_aware_fak_estimate(self) -> None:
        client = FakeClient()

        price = PolymarketGateway(client).get_executable_sell_price(
            token_id="token-1",
            shares=Decimal("7.5"),
        )

        self.assertEqual(price, Decimal("0.18"))
        self.assertEqual(
            client.estimate_calls,
            [{
                "token_id": "token-1",
                "side": "SELL",
                "shares": Decimal("7.5"),
                "order_type": "FAK",
            }],
        )

    def test_redeem_waits_for_terminal_transaction(self) -> None:
        client = FakeClient()

        transaction_hash = PolymarketGateway(client).redeem(
            condition_id="condition-1"
        )

        self.assertEqual(transaction_hash, "0xtransaction")
        self.assertEqual(client.redeem_calls, [{"condition_id": "condition-1"}])
        self.assertTrue(client.transaction_handle.waited)


class ReconciliationTests(unittest.TestCase):
    def test_entry_uses_exchange_shares_and_average_price(self) -> None:
        local = {
            "status": "pending",
            "shares": 0.0,
            "entry_price": 0.30,
            "fee_rate": 0.05,
        }
        exchange = PositionSnapshot(
            token_id="token-1",
            condition_id="condition-1",
            shares=Decimal("7.5"),
            average_price=Decimal("0.22"),
        )

        reconcile_entry(local, exchange)

        self.assertEqual(local["status"], "open")
        self.assertEqual(local["shares"], 7.5)
        self.assertEqual(local["entry_price"], 0.22)
        self.assertEqual(local["entry_fee"], 0.06435)
        self.assertEqual(local["amount"], 1.71435)

    def test_entry_without_exchange_position_stays_pending(self) -> None:
        local = {"status": "pending", "shares": 0.0}

        reconcile_entry(local, None)

        self.assertEqual(local["status"], "pending")
        self.assertEqual(local["shares"], 0.0)

    def test_entry_preserves_resolution_metadata(self) -> None:
        local = {"status": "pending", "shares": 0.0}
        exchange = PositionSnapshot(
            token_id="token-1",
            condition_id="condition-1",
            shares=Decimal("4"),
            average_price=Decimal("0.25"),
            redeemable=True,
            current_price=Decimal("1"),
        )

        reconcile_entry(local, exchange)

        self.assertTrue(local["redeemable"])
        self.assertEqual(local["current_price"], 1.0)

    def test_exit_only_closes_when_exchange_position_is_gone(self) -> None:
        local = {"status": "exit_pending", "shares": 7.5, "entry_price": 0.22}

        closed = reconcile_exit(local, None, exit_price=Decimal("0.40"))

        self.assertTrue(closed)
        self.assertEqual(local["status"], "closed")
        self.assertEqual(local["shares"], 0.0)
        self.assertEqual(local["pnl"], 1.35)

    def test_partial_exit_keeps_residual_position_open(self) -> None:
        local = {"status": "exit_pending", "shares": 7.5, "entry_price": 0.22}
        exchange = PositionSnapshot(
            token_id="token-1",
            condition_id="condition-1",
            shares=Decimal("2.0"),
            average_price=Decimal("0.22"),
        )

        closed = reconcile_exit(local, exchange, exit_price=Decimal("0.40"))

        self.assertFalse(closed)
        self.assertEqual(local["status"], "open")
        self.assertEqual(local["shares"], 2.0)

    def test_exit_pnl_deducts_entry_and_exit_taker_fees(self) -> None:
        local = {
            "status": "exit_pending",
            "shares": 10.0,
            "entry_price": 0.25,
            "fee_rate": 0.05,
        }

        reconcile_exit(local, None, exit_price=Decimal("0.40"))

        self.assertEqual(local["pnl"], 1.29)

    def test_unmanaged_exchange_position_is_recorded_and_reported(self) -> None:
        state = {"positions": []}
        exchange = {
            "token-1": PositionSnapshot(
                token_id="token-1",
                condition_id="condition-1",
                shares=Decimal("7.5"),
                average_price=Decimal("0.22"),
            )
        }

        report = reconcile_state(state, exchange)

        self.assertEqual(report.unmanaged_tokens, ("token-1",))
        self.assertEqual(state["positions"][0]["status"], "unmanaged")
        self.assertEqual(state["positions"][0]["shares"], 7.5)

        second_report = reconcile_state(state, exchange)
        self.assertEqual(second_report.unmanaged_tokens, ("token-1",))

    def test_missing_local_position_is_not_silently_closed(self) -> None:
        state = {
            "positions": [
                {"token_id": "token-1", "status": "open", "shares": 7.5}
            ]
        }

        report = reconcile_state(state, {})

        self.assertEqual(report.missing_tokens, ("token-1",))
        self.assertEqual(state["positions"][0]["status"], "missing")

    def test_exchange_position_behind_closed_record_is_unmanaged(self) -> None:
        state = {
            "positions": [
                {"token_id": "token-1", "status": "closed", "shares": 0.0}
            ]
        }
        exchange = {
            "token-1": PositionSnapshot(
                token_id="token-1",
                condition_id="condition-1",
                shares=Decimal("2"),
                average_price=Decimal("0.25"),
            )
        }

        report = reconcile_state(state, exchange)

        self.assertEqual(report.unmanaged_tokens, ("token-1",))
        self.assertEqual(state["positions"][0]["status"], "unmanaged")

    def test_pending_entry_is_filled_from_exchange_state(self) -> None:
        state = {
            "positions": [
                {
                    "token_id": "token-1",
                    "condition_id": "condition-1",
                    "status": "pending",
                    "shares": 0.0,
                    "entry_price": 0.30,
                }
            ]
        }
        exchange = {
            "token-1": PositionSnapshot(
                token_id="token-1",
                condition_id="condition-1",
                shares=Decimal("4"),
                average_price=Decimal("0.25"),
            )
        }

        report = reconcile_state(state, exchange)

        self.assertFalse(report.has_discrepancies)
        self.assertEqual(state["positions"][0]["status"], "open")
        self.assertEqual(state["positions"][0]["shares"], 4.0)

    def test_unconfirmed_exit_does_not_submit_again(self) -> None:
        state = {
            "positions": [
                {
                    "token_id": "token-1",
                    "status": "exit_pending",
                    "shares": 7.5,
                    "shares_before_exit": 7.5,
                    "entry_price": 0.22,
                    "exit_price_requested": 0.40,
                }
            ]
        }
        exchange = {
            "token-1": PositionSnapshot(
                token_id="token-1",
                condition_id="condition-1",
                shares=Decimal("7.5"),
                average_price=Decimal("0.22"),
            )
        }

        reconcile_state(state, exchange)

        self.assertEqual(state["positions"][0]["status"], "exit_pending")

    def test_exit_requires_two_absent_snapshots_before_closing(self) -> None:
        state = {
            "positions": [
                {
                    "token_id": "token-1",
                    "status": "exit_pending",
                    "shares": 7.5,
                    "shares_before_exit": 7.5,
                    "entry_price": 0.22,
                    "exit_price_requested": 0.40,
                }
            ]
        }

        reconcile_state(state, {})
        self.assertEqual(state["positions"][0]["status"], "exit_pending")

        reconcile_state(state, {})
        self.assertEqual(state["positions"][0]["status"], "closed")


class ProcessLockTests(unittest.TestCase):
    def test_second_process_lock_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "live.lock"
            first = ProcessLock(path)
            second = ProcessLock(path)
            first.acquire()
            try:
                with self.assertRaises(ProcessLockError):
                    second.acquire()
            finally:
                first.release()


if __name__ == "__main__":
    unittest.main()
