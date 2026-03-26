"""
Anti-Gravity Financial Agentic System
Deterministic Scoring Engine — FinancialEngine class
Calculates 5 financial health scores (0-10) from raw user data.
"""

from __future__ import annotations
from typing import Dict


# Fidelity age-based retirement savings multipliers
# Maps age => recommended multiple of annual salary saved
FIDELITY_MULTIPLIERS = {
    25: 0.5,
    30: 1.0,
    35: 2.0,
    40: 3.0,
    45: 4.0,
    50: 6.0,
    55: 7.0,
    60: 8.0,
    65: 10.0,
    67: 10.0,
}


def _get_fidelity_target(age: int) -> float:
    """Return the Fidelity recommended savings multiple for a given age."""
    if age <= 25:
        return FIDELITY_MULTIPLIERS[25]
    if age >= 67:
        return FIDELITY_MULTIPLIERS[67]
    # Linear interpolation between known milestones
    milestones = sorted(FIDELITY_MULTIPLIERS.keys())
    for i in range(len(milestones) - 1):
        lower, upper = milestones[i], milestones[i + 1]
        if lower <= age <= upper:
            frac = (age - lower) / (upper - lower)
            return FIDELITY_MULTIPLIERS[lower] + frac * (
                FIDELITY_MULTIPLIERS[upper] - FIDELITY_MULTIPLIERS[lower]
            )
    return FIDELITY_MULTIPLIERS[67]


def _clamp(value: float, lo: float = 0.0, hi: float = 10.0) -> float:
    return max(lo, min(hi, value))


class FinancialEngine:
    """
    Deterministic scoring engine.
    Accepts a UserData dict and produces six 0-10 scores.
    """

    def __init__(self, user_data: dict):
        self.data = user_data

    # ------------------------------------------------------------------
    # Individual score methods
    # ------------------------------------------------------------------

    def calculate_emergency_score(self) -> float:
        """
        Emergency Fund: min(10, (liquid_savings / (essential_expenses * 6)) * 10)
        Target: 6 months of essential expenses.
        """
        expenses = self.data.get("essential_expenses", 1)
        savings = self.data.get("liquid_savings", 0)
        if expenses <= 0:
            return 10.0  # no expenses => no need
        raw = (savings / (expenses * 6)) * 10
        return round(_clamp(raw), 2)

    def calculate_debt_score(self) -> float:
        """
        Debt Score: 10 - (total_monthly_debt / monthly_income * 20)
        If any interest rate > 7%, subtract 4 additional points.
        """
        income = self.data.get("monthly_income", 1)
        monthly_debt = self.data.get("monthly_debt_payment", 0)
        rates = self.data.get("debt_interest_rates", [])

        if income <= 0:
            return 0.0

        raw = 10 - (monthly_debt / income * 20)

        # High-interest penalty
        if any(r > 7 for r in rates):
            raw -= 4

        return round(_clamp(raw), 2)

    def calculate_insurance_score(self) -> float:
        """
        Insurance Score: 10 points, deduct for missing coverage.
        """
        status = self.data.get("insurance_status", {})
        score = 10.0
        if not status.get("health", False):
            score -= 5
        if not status.get("life", False):
            score -= 3
        if not status.get("disability", False):
            score -= 2
        return round(_clamp(score), 2)

    def calculate_retirement_score(self) -> float:
        """
        Retirement Score based on Fidelity age multipliers.
        Compares actual retirement savings to target.
        """
        age = self.data.get("age", 30)
        annual_income = self.data.get("annual_income", 0)
        if annual_income <= 0:
            annual_income = self.data.get("monthly_income", 0) * 12
        retirement = self.data.get("retirement_savings", 0)

        target_multiple = _get_fidelity_target(age)
        target_amount = annual_income * target_multiple
        if target_amount <= 0:
            return 10.0

        raw = (retirement / target_amount) * 10
        return round(_clamp(raw), 2)


    def calculate_investment_score(self) -> float:
        """
        Investment Score: 100-Age Rule check.
        Target equity %  = (100 - age).
        """
        age = self.data.get("age", 30)
        current_eq = self.data.get("current_equity_pct", 0)
        target_eq = 100 - age

        if target_eq <= 0:
            return 10.0

        deviation = abs(current_eq - target_eq)
        # Perfect allocation → 10; 50pt deviation → 0
        raw = 10 - (deviation / 5)
        return round(_clamp(raw), 2)

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------

    def calculate_all_scores(self) -> Dict[str, float]:
        """Return a dict of all five scores."""
        return {
            "emergency": self.calculate_emergency_score(),
            "debt": self.calculate_debt_score(),
            "insurance": self.calculate_insurance_score(),
            "retirement": self.calculate_retirement_score(),
            "investment": self.calculate_investment_score(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_emergency_gap(self) -> float:
        """Dollar amount still needed to reach 6-month emergency fund."""
        expenses = self.data.get("essential_expenses", 0)
        savings = self.data.get("liquid_savings", 0)
        target = expenses * 6
        return max(0, target - savings)

    def get_ideal_equity_pct(self) -> float:
        """Target equity allocation from 100-Age rule."""
        return 100 - self.data.get("age", 30)
