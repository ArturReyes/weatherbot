"""Exchange-facing primitives for safe live trading."""

from __future__ import annotations

import fcntl
import os
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, BinaryIO


@dataclass(frozen=True)
class OrderSubmission:
    accepted: bool
    order_id: str | None
    status: str
    reason: str | None


@dataclass(frozen=True)
class PositionSnapshot:
    token_id: str
    condition_id: str
    shares: Decimal
    average_price: Decimal
    redeemable: bool = False
    current_price: Decimal = Decimal("0")


@dataclass(frozen=True)
class ReconciliationReport:
    unmanaged_tokens: tuple[str, ...] = ()
    missing_tokens: tuple[str, ...] = ()

    @property
    def has_discrepancies(self) -> bool:
        return bool(self.unmanaged_tokens or self.missing_tokens)


class PolymarketGateway:
    def __init__(self, client: Any) -> None:
        self._client = client

    def get_balance(self) -> Decimal:
        result = self._client.get_balance_allowance(asset_type="COLLATERAL")
        return Decimal(str(result.balance))

    def buy(
        self,
        *,
        token_id: str,
        amount: Decimal,
        max_price: Decimal,
    ) -> OrderSubmission:
        response = self._client.place_market_order(
            token_id=token_id,
            side="BUY",
            amount=amount,
            max_price=max_price,
            order_type="FAK",
        )
        return _normalize_order_submission(response)

    def get_positions(self) -> dict[str, PositionSnapshot]:
        result: dict[str, PositionSnapshot] = {}
        paginator = self._client.list_positions()
        for position in paginator.iter_items():
            if position.token_id is None or position.size is None:
                continue
            token_id = str(position.token_id)
            result[token_id] = PositionSnapshot(
                token_id=token_id,
                condition_id=str(position.condition_id),
                shares=Decimal(str(position.size)),
                average_price=Decimal(str(position.avg_price or 0)),
                redeemable=bool(position.redeemable),
                current_price=Decimal(str(position.cur_price or 0)),
            )
        return result

    def get_executable_sell_price(
        self,
        *,
        token_id: str,
        shares: Decimal,
    ) -> Decimal:
        return Decimal(
            str(
                self._client.estimate_market_price(
                    token_id=token_id,
                    side="SELL",
                    shares=shares,
                    order_type="FAK",
                )
            )
        )

    def redeem(self, *, condition_id: str) -> str:
        handle = self._client.redeem_positions(condition_id=condition_id)
        outcome = handle.wait()
        return str(outcome.transaction_hash)

    def sell(
        self,
        *,
        token_id: str,
        shares: Decimal,
        min_price: Decimal,
    ) -> OrderSubmission:
        response = self._client.place_market_order(
            token_id=token_id,
            side="SELL",
            shares=shares,
            min_price=min_price,
            order_type="FAK",
        )
        return _normalize_order_submission(response)


def _normalize_order_submission(response: Any) -> OrderSubmission:
    if getattr(response, "ok", False):
        return OrderSubmission(
            accepted=True,
            order_id=str(response.order_id),
            status=str(response.status).lower(),
            reason=None,
        )
    code = str(getattr(response, "code", "unknown"))
    message = str(getattr(response, "message", "Unknown order rejection"))
    return OrderSubmission(False, None, "rejected", f"{code}: {message}")


def reconcile_entry(
    local: dict[str, Any],
    exchange: PositionSnapshot | None,
) -> None:
    if exchange is None:
        local["status"] = "pending"
        return
    local["status"] = "open"
    local["shares"] = float(exchange.shares)
    local["entry_price"] = float(exchange.average_price)
    fee_rate = Decimal(str(local.get("fee_rate", 0)))
    entry_fee = _trade_fee(exchange.shares, exchange.average_price, fee_rate)
    local["entry_fee"] = round(float(entry_fee), 5)
    local["amount"] = round(
        float(exchange.shares * exchange.average_price + entry_fee),
        5,
    )
    local["redeemable"] = exchange.redeemable
    local["current_price"] = float(exchange.current_price)


def reconcile_exit(
    local: dict[str, Any],
    exchange: PositionSnapshot | None,
    *,
    exit_price: Decimal,
) -> bool:
    shares_before = Decimal(str(local.get("shares", 0)))
    shares_after = exchange.shares if exchange is not None else Decimal("0")
    sold_shares = max(Decimal("0"), shares_before - shares_after)
    entry_price = Decimal(str(local.get("entry_price", 0)))
    fee_rate = Decimal(str(local.get("fee_rate", 0)))
    entry_fee = _trade_fee(sold_shares, entry_price, fee_rate)
    exit_fee = _trade_fee(sold_shares, exit_price, fee_rate)
    realized = (exit_price - entry_price) * sold_shares - entry_fee - exit_fee
    local["exit_fees"] = round(
        float(Decimal(str(local.get("exit_fees", 0))) + exit_fee),
        5,
    )
    local["realized_pnl"] = round(
        float(Decimal(str(local.get("realized_pnl", 0))) + realized),
        2,
    )
    local["shares"] = float(shares_after)
    if exchange is not None and shares_after > 0:
        local["status"] = "open"
        return False
    local["status"] = "closed"
    local["pnl"] = local["realized_pnl"]
    return True


def _trade_fee(shares: Decimal, price: Decimal, fee_rate: Decimal) -> Decimal:
    if shares <= 0 or price <= 0 or price >= 1 or fee_rate <= 0:
        return Decimal("0")
    return shares * fee_rate * price * (Decimal("1") - price)


def reconcile_state(
    state: dict[str, Any],
    exchange: dict[str, PositionSnapshot],
) -> ReconciliationReport:
    local_positions = state.setdefault("positions", [])
    local_by_token = {
        str(position["token_id"]): position
        for position in local_positions
        if position.get("token_id")
    }
    missing: list[str] = []
    unmanaged_existing: list[str] = []

    for token_id, local in local_by_token.items():
        status = local.get("status")
        snapshot = exchange.get(token_id)

        if status in {"submitting", "pending", "unknown"}:
            if snapshot is not None:
                reconcile_entry(local, snapshot)
            continue

        if status in {"redeeming", "redemption_unknown", "redemption_confirmed"}:
            if snapshot is None and status == "redemption_confirmed":
                local["status"] = "redeemed"
                local["shares"] = 0.0
            elif snapshot is not None:
                local["redeemable"] = snapshot.redeemable
                local["current_price"] = float(snapshot.current_price)
            continue

        if status in {"exit_pending", "exit_unknown"}:
            if snapshot is None:
                confirmations = int(local.get("exit_absence_confirmations", 0)) + 1
                local["exit_absence_confirmations"] = confirmations
                if confirmations < 2:
                    continue
                reconcile_exit(
                    local,
                    None,
                    exit_price=Decimal(str(local.get("exit_price_requested", 0))),
                )
                continue
            before = Decimal(str(local.get("shares_before_exit", local.get("shares", 0))))
            if snapshot.shares < before:
                reconcile_exit(
                    local,
                    snapshot,
                    exit_price=Decimal(str(local.get("exit_price_requested", 0))),
                )
            continue

        if status in {"open", "missing"}:
            if snapshot is None:
                local["status"] = "missing"
                missing.append(token_id)
            else:
                reconcile_entry(local, snapshot)
            continue

        if status == "unmanaged" and snapshot is not None:
            local["shares"] = float(snapshot.shares)
            local["entry_price"] = float(snapshot.average_price)
            unmanaged_existing.append(token_id)
            continue

        if snapshot is not None:
            local["status"] = "unmanaged"
            local["shares"] = float(snapshot.shares)
            local["entry_price"] = float(snapshot.average_price)
            unmanaged_existing.append(token_id)

    unmanaged_new = sorted(set(exchange) - set(local_by_token))
    for token_id in unmanaged_new:
        snapshot = exchange[token_id]
        local_positions.append(
            {
                "token_id": token_id,
                "condition_id": snapshot.condition_id,
                "status": "unmanaged",
                "shares": float(snapshot.shares),
                "entry_price": float(snapshot.average_price),
                "amount": round(float(snapshot.shares * snapshot.average_price), 2),
                "redeemable": snapshot.redeemable,
                "current_price": float(snapshot.current_price),
            }
        )

    return ReconciliationReport(
        unmanaged_tokens=tuple(sorted(unmanaged_existing + unmanaged_new)),
        missing_tokens=tuple(sorted(missing)),
    )


class ProcessLockError(RuntimeError):
    pass


class ProcessLock:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle: BinaryIO | None = None

    def acquire(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        handle = self._path.open("a+b")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            handle.close()
            raise ProcessLockError(
                f"Another live-trading process holds {self._path}"
            ) from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()).encode("ascii"))
        handle.flush()
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def __enter__(self) -> "ProcessLock":
        self.acquire()
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
