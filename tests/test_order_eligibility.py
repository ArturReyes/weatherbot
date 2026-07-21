from __future__ import annotations

import unittest

from order_eligibility import evaluate_order_eligibility


class OrderEligibilityTests(unittest.TestCase):
    def test_exchange_minimum_can_exceed_operator_floor(self) -> None:
        decision = evaluate_order_eligibility(
            proposed_notional=0.19,
            entry_price=0.11,
            fee_rate=0.05,
            min_order_size=5,
            min_trade_notional=0.50,
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "below_order_minimum")
        self.assertGreater(decision.required_notional, 0.55)
        self.assertEqual(decision.proposed_notional, 0.19)

    def test_eligible_order_is_not_upsized(self) -> None:
        decision = evaluate_order_eligibility(
            proposed_notional=0.60,
            entry_price=0.11,
            fee_rate=0.05,
            min_order_size=5,
            min_trade_notional=0.50,
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.proposed_notional, 0.60)
        self.assertAlmostEqual(decision.shares, 0.60 / 0.114895)

    def test_missing_exchange_minimum_fails_closed(self) -> None:
        decision = evaluate_order_eligibility(
            proposed_notional=5.0,
            entry_price=0.20,
            fee_rate=0.0,
            min_order_size=None,
            min_trade_notional=0.50,
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "missing_min_order_size")
