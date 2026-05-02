"""
Portfolio Risk Calculator
Computes key risk metrics for a wealth manager across crash scenarios.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import TypedDict

# Type definitions
class AssetDict(TypedDict):
    name: str
    allocation_pct: float          # 0-100
    expected_crash_pct: float      # negative value, e.g. -80 means -80 %

class PortfolioDict(TypedDict):
    total_value_inr: float
    monthly_expenses_inr: float
    assets: list[AssetDict]

class ScenarioResult(TypedDict):
    post_crash_value: float
    runway_months: float
    ruin_test: str
    largest_risk_asset: str
    concentration_warning: bool

class RiskMetrics(TypedDict):
    severe: ScenarioResult
    moderate: ScenarioResult


# Helper functions
def _validate_portfolio(portfolio: PortfolioDict) -> None:
    """Raise ValueError for structurally invalid portfolios."""
    total_value = portfolio.get("total_value_inr", 0)
    if total_value < 0:
        raise ValueError("total_value_inr must be non-negative.")

    monthly_expenses = portfolio.get("monthly_expenses_inr", 0)
    if monthly_expenses < 0:
        raise ValueError("monthly_expenses_inr must be non-negative.")

    assets = portfolio.get("assets", [])
    total_allocation = sum(a["allocation_pct"] for a in assets)
    if assets and abs(total_allocation - 100) > 0.01:
        raise ValueError(
            f"Asset allocations sum to {total_allocation:.2f}%, expected 100%."
        )


def _asset_value(total: float, allocation_pct: float) -> float:
    """Return the INR value of an asset given its allocation percentage."""
    return total * (allocation_pct / 100)


def _post_crash_value(
    total_value: float,
    assets: list[AssetDict],
    crash_multiplier: float = 1.0,
) -> float:
    """
    Compute total portfolio value after applying crash scenarios.

    crash_multiplier=1.0  → severe  (full expected crash)
    crash_multiplier=0.5  → moderate (50 % of expected crash)
    """
    surviving_value = 0.0
    for asset in assets:
        value = _asset_value(total_value, asset["allocation_pct"])
        loss_pct = asset["expected_crash_pct"] * crash_multiplier   # already negative
        surviving_value += value * (1 + loss_pct / 100)
    return max(surviving_value, 0.0)


def _runway_months(post_crash_value: float, monthly_expenses: float) -> float:
    """Months the portfolio can sustain expenses. Infinite if expenses are zero."""
    if monthly_expenses <= 0:
        return float("inf")
    return post_crash_value / monthly_expenses


def _ruin_test(runway: float, threshold_months: int = 12) -> str:
    return "PASS" if runway > threshold_months else "FAIL"


def _largest_risk_asset(assets: list[AssetDict]) -> str:
    """
    Return the name of the asset with the highest risk weight:
        risk_weight = allocation_pct × |crash_pct|
    """
    if not assets:
        return "N/A"
    return max(
        assets,
        key=lambda a: a["allocation_pct"] * abs(a["expected_crash_pct"]),
    )["name"]


def _concentration_warning(assets: list[AssetDict], threshold_pct: float = 40.0) -> bool:
    """True if any single asset exceeds the concentration threshold."""
    return any(a["allocation_pct"] > threshold_pct for a in assets)


def _build_scenario(
    total_value: float,
    monthly_expenses: float,
    assets: list[AssetDict],
    crash_multiplier: float,
) -> ScenarioResult:
    pcv = _post_crash_value(total_value, assets, crash_multiplier)
    runway = _runway_months(pcv, monthly_expenses)
    return ScenarioResult(
        post_crash_value=round(pcv, 2),
        runway_months=round(runway, 2),
        ruin_test=_ruin_test(runway),
        largest_risk_asset=_largest_risk_asset(assets),
        concentration_warning=_concentration_warning(assets),
    )


# Public API
def compute_risk_metrics(portfolio: PortfolioDict) -> RiskMetrics:
    """
    Compute portfolio risk metrics for two crash scenarios.

    Returns
    -------
    RiskMetrics
        - severe   : full expected crash applied to every asset
        - moderate : each asset loses 50 % of its expected crash magnitude
    """
    _validate_portfolio(portfolio)

    total_value     = portfolio["total_value_inr"]
    monthly_expenses = portfolio["monthly_expenses_inr"]
    assets          = portfolio["assets"]

    return RiskMetrics(
        severe=_build_scenario(total_value, monthly_expenses, assets, crash_multiplier=1.0),
        moderate=_build_scenario(total_value, monthly_expenses, assets, crash_multiplier=0.5),
    )

# CLI visualisation helpers
BAR_WIDTH = 40          # maximum bar characters


def _bar(pct: float, width: int = BAR_WIDTH) -> str:
    """Return a filled bar proportional to pct (0-100)."""
    filled = round(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def print_allocation_chart(portfolio: PortfolioDict) -> None:
    """Print a horizontal bar chart of asset allocations to the terminal."""
    total_value = portfolio["total_value_inr"]
    assets = portfolio["assets"]

    print("\n" + "═" * 65)
    print("  PORTFOLIO ALLOCATION BREAKDOWN")
    print("═" * 65)
    print(f"  {'Asset':<12} {'Alloc %':>7}  {'Bar':<{BAR_WIDTH}}  {'Value (INR)':>14}")
    print("─" * 65)

    for asset in assets:
        pct   = asset["allocation_pct"]
        value = _asset_value(total_value, pct)
        bar   = _bar(pct)
        print(f"  {asset['name']:<12} {pct:>6.1f}%  {bar}  {value:>14,.0f}")

    print("═" * 65)


def print_risk_report(portfolio: PortfolioDict) -> None:
    """Print a full risk report including both crash scenarios."""
    metrics = compute_risk_metrics(portfolio)

    print_allocation_chart(portfolio)

    print("\n" + "═" * 65)
    print("  CRASH SCENARIO ANALYSIS")
    print("═" * 65)

    col_w = 22
    label_w = 26
    print(f"  {'Metric':<{label_w}} {'Severe Crash':>{col_w}} {'Moderate Crash':>{col_w}}")
    print("─" * 65)

    def _fmt_inr(v: float) -> str:
        if v == float("inf"):
            return "∞"
        return f"₹{v:,.0f}"

    def _fmt_months(v: float) -> str:
        if v == float("inf"):
            return "∞ months"
        return f"{v:.1f} months"

    rows = [
        ("Post-crash Value",
         _fmt_inr(metrics["severe"]["post_crash_value"]),
         _fmt_inr(metrics["moderate"]["post_crash_value"])),
        ("Runway",
         _fmt_months(metrics["severe"]["runway_months"]),
         _fmt_months(metrics["moderate"]["runway_months"])),
        ("Ruin Test (>12 mo)",
         metrics["severe"]["ruin_test"],
         metrics["moderate"]["ruin_test"]),
        ("Largest Risk Asset",
         metrics["severe"]["largest_risk_asset"],
         metrics["moderate"]["largest_risk_asset"]),
        ("Concentration Warning",
         str(metrics["severe"]["concentration_warning"]),
         str(metrics["moderate"]["concentration_warning"])),
    ]

    for label, severe_val, moderate_val in rows:
        print(f"  {label:<{label_w}} {severe_val:>{col_w}} {moderate_val:>{col_w}}")

    print("═" * 65)
    print()


# Entry point

if __name__ == "__main__":
    portfolio: PortfolioDict = {
        "total_value_inr": 10_000_000,
        "monthly_expenses_inr": 80_000,
        "assets": [
            {"name": "BTC",     "allocation_pct": 30, "expected_crash_pct": -80},
            {"name": "NIFTY50", "allocation_pct": 40, "expected_crash_pct": -40},
            {"name": "GOLD",    "allocation_pct": 20, "expected_crash_pct": -15},
            {"name": "CASH",    "allocation_pct": 10, "expected_crash_pct":   0},
        ],
    }

    print_risk_report(portfolio)

    # Edge-case demos
    print("── Edge cases ──────────────────────────────────────────────────")

    zero_expense = {**portfolio, "monthly_expenses_inr": 0}
    m = compute_risk_metrics(zero_expense)
    print(f"  Zero expenses → runway (severe): {m['severe']['runway_months']}")

    all_cash: PortfolioDict = {
        "total_value_inr": 10_000_000,
        "monthly_expenses_inr": 80_000,
        "assets": [{"name": "CASH", "allocation_pct": 100, "expected_crash_pct": 0}],
    }
    m2 = compute_risk_metrics(all_cash)
    print(f"  100 % cash → post-crash value (severe): ₹{m2['severe']['post_crash_value']:,.0f}")
    print(f"  100 % cash → runway: {m2['severe']['runway_months']:.1f} months")
    print()
