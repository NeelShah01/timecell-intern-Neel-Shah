"""
╔══════════════════════════════════════════════════════════════════════╗
║          MONTE CARLO WEALTH RUNWAY SIMULATOR                         ║
║     A probabilistic financial planning tool for Indian investors     ║
║                                                                      ║
║  What most wealth apps won't tell you: the ORDER of returns matters. ║
║  A 12% CAGR that arrives as [+50%, -40%, +40%] destroys more         ║
║  wealth than the average suggests. This tool simulates 10,000        ║
║  possible futures — not just the expected one.                       ║
╚══════════════════════════════════════════════════════════════════════╝

Usage
-----
    python monte_carlo.py                         # default portfolio, drawdown mode
    python monte_carlo.py --mode accumulation     # building wealth (with monthly savings)
    python monte_carlo.py --years 30              # 30-year horizon
    python monte_carlo.py --sims 5000             # fewer simulations, faster
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass, field
from typing import Any

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42

# ── Terminal colours (ANSI) ───────────────────────────────────────────────────
C_RESET  = "\033[0m"
C_BOLD   = "\033[1m"
C_DIM    = "\033[2m"
C_RED    = "\033[91m"
C_GREEN  = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN   = "\033[96m"
C_WHITE  = "\033[97m"
C_GRAY   = "\033[90m"


# ═══════════════════════════════════════════════════════════════════════════════
# 1. ASSET CLASS PARAMETERS  (Indian market calibrated)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class AssetParams:
    """
    Annual return parameters for a single asset class.

    We model returns as log-normal:
        annual_return = exp(μ + σ·Z) - 1,  Z ~ N(0,1)

    The μ and σ for the log-normal are derived from the arithmetic
    mean and standard deviation of the annual returns.
    """
    name:         str
    mean_return:  float   # arithmetic mean annual return (e.g. 0.12 = 12%)
    annual_std:   float   # standard deviation of annual returns
    currency:     str = "INR"

    @property
    def lognormal_mu(self) -> float:
        """Convert arithmetic mean/std to log-normal μ."""
        variance = self.annual_std ** 2
        mean_r   = 1 + self.mean_return
        return math.log(mean_r ** 2 / math.sqrt(mean_r ** 2 + variance))

    @property
    def lognormal_sigma(self) -> float:
        """Convert arithmetic mean/std to log-normal σ."""
        variance = self.annual_std ** 2
        mean_r   = 1 + self.mean_return
        return math.sqrt(math.log(1 + variance / mean_r ** 2))

    def sample_return(self, rng: random.Random) -> float:
        """Draw one annual return from the log-normal distribution."""
        z = rng.gauss(0, 1)
        return math.exp(self.lognormal_mu + self.lognormal_sigma * z) - 1


# Indian market calibrated parameters (2010-2024 data informed)
ASSET_PARAMS: dict[str, AssetParams] = {
    "BTC":     AssetParams("Bitcoin",        mean_return=0.60,  annual_std=0.85, currency="USD"),
    "NIFTY50": AssetParams("NIFTY 50",       mean_return=0.12,  annual_std=0.22, currency="INR"),
    "SENSEX":  AssetParams("SENSEX",         mean_return=0.12,  annual_std=0.21, currency="INR"),
    "GOLD":    AssetParams("Gold",           mean_return=0.08,  annual_std=0.13, currency="INR"),
    "BONDS":   AssetParams("Govt Bonds",     mean_return=0.07,  annual_std=0.04, currency="INR"),
    "FD":      AssetParams("Fixed Deposit",  mean_return=0.065, annual_std=0.01, currency="INR"),
    "CASH":    AssetParams("Cash",           mean_return=0.055, annual_std=0.00, currency="INR"),
    "REALTY":  AssetParams("Real Estate",    mean_return=0.09,  annual_std=0.10, currency="INR"),
    "USDEBT":  AssetParams("US Bonds",       mean_return=0.045, annual_std=0.06, currency="USD"),
}

INDIAN_INFLATION_MEAN = 0.060   # 6% annual CPI
INDIAN_INFLATION_STD  = 0.020   # inflation itself is uncertain

INR_USD_DEPRECIATION_MEAN = 0.035   # INR depreciates ~3.5% vs USD per year historically
INR_USD_DEPRECIATION_STD  = 0.06


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PORTFOLIO MODEL
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PortfolioConfig:
    """
    Full portfolio specification for simulation.

    Parameters
    ----------
    total_value_inr     : Current portfolio value in INR.
    monthly_expenses_inr: Monthly living expenses (will grow with inflation).
    monthly_savings_inr : Monthly savings/contributions (0 = pure drawdown mode).
    assets              : List of {"name": str, "allocation_pct": float} dicts.
    """
    total_value_inr:      float
    monthly_expenses_inr: float
    monthly_savings_inr:  float
    assets:               list[dict[str, Any]]

    def allocation_map(self) -> dict[str, float]:
        """Return {asset_name: fraction} map, validated to sum to 1."""
        m = {a["name"]: a["allocation_pct"] / 100 for a in self.assets}
        total = sum(m.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Allocations sum to {total*100:.1f}%, expected 100%.")
        return m


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_single_path(
    config:   PortfolioConfig,
    n_years:  int,
    rng:      random.Random,
) -> list[float]:
    """
    Simulate one possible wealth trajectory over n_years.

    Each year:
      1. Each asset class generates its return (log-normal draw).
      2. Portfolio grows by the weighted average return.
      3. INR/USD rate is applied to USD-denominated assets.
      4. Inflation is drawn and applied to expenses.
      5. Net cashflow (savings - expenses) is added.
      6. Wealth floor at 0 (ruin = stays at 0).

    Returns
    -------
    List of portfolio values at end of each year (length = n_years + 1,
    index 0 is the starting value).
    """
    allocation   = config.allocation_map()
    wealth       = config.total_value_inr
    expenses_inr = config.monthly_expenses_inr * 12      # annualise
    savings_inr  = config.monthly_savings_inr  * 12

    path = [wealth]

    for _ in range(n_years):
        if wealth <= 0:
            path.append(0.0)
            continue

        # ── Draw this year's macro conditions ────────────────────────────
        inflation   = max(0, rng.gauss(INDIAN_INFLATION_MEAN, INDIAN_INFLATION_STD))
        inr_deprec  = rng.gauss(INR_USD_DEPRECIATION_MEAN, INR_USD_DEPRECIATION_STD)

        # ── Compute portfolio return ──────────────────────────────────────
        portfolio_return = 0.0
        for asset_name, weight in allocation.items():
            params = ASSET_PARAMS.get(asset_name)
            if params is None:
                # Unknown asset: treat as cash
                params = ASSET_PARAMS["CASH"]

            asset_return = params.sample_return(rng)

            # Convert USD-denominated assets back to INR
            if params.currency == "USD":
                asset_return = (1 + asset_return) * (1 + inr_deprec) - 1

            portfolio_return += weight * asset_return

        # ── Update wealth ─────────────────────────────────────────────────
        wealth = wealth * (1 + portfolio_return)

        # ── Net cashflow (savings in, expenses out) ───────────────────────
        wealth += savings_inr - expenses_inr

        # ── Grow expenses with inflation ──────────────────────────────────
        expenses_inr *= (1 + inflation)

        # ── Ruin floor ────────────────────────────────────────────────────
        wealth = max(wealth, 0.0)
        path.append(wealth)

    return path


def run_simulation(
    config:       PortfolioConfig,
    n_years:      int,
    n_simulations: int,
) -> list[list[float]]:
    """Run n_simulations independent wealth paths. Returns list of paths."""
    rng = random.Random(SEED)
    return [
        simulate_single_path(config, n_years, rng)
        for _ in range(n_simulations)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════

def percentile(values: list[float], pct: float) -> float:
    """Return the p-th percentile (0-100) of a sorted or unsorted list."""
    if not values:
        return 0.0
    sv = sorted(values)
    k  = (len(sv) - 1) * pct / 100
    lo, hi = int(k), min(int(k) + 1, len(sv) - 1)
    frac = k - lo
    return sv[lo] + frac * (sv[hi] - sv[lo])


def extract_year_slices(paths: list[list[float]], n_years: int) -> list[list[float]]:
    """
    Transpose paths into year slices.
    year_slices[t] = list of wealth values across all simulations at year t.
    """
    return [[path[t] for path in paths] for t in range(n_years + 1)]


def ruin_probability_by_year(paths: list[list[float]], n_years: int) -> list[float]:
    """
    For each year t, return the fraction of simulations where
    the portfolio has hit zero at some point on or before year t.
    """
    n = len(paths)
    ruined_by = [False] * n    # has simulation i ever ruined?
    result = []
    for t in range(n_years + 1):
        for i, path in enumerate(paths):
            if path[t] <= 0:
                ruined_by[i] = True
        result.append(sum(ruined_by) / n)
    return result


def cagr_from_path(path: list[float]) -> float:
    """Compound Annual Growth Rate of a single path."""
    years = len(path) - 1
    if years <= 0 or path[0] <= 0 or path[-1] <= 0:
        return 0.0
    return (path[-1] / path[0]) ** (1 / years) - 1


def find_fire_year(
    paths:         list[list[float]],
    n_years:       int,
    fire_multiple: float = 25.0,    # 4% rule → 25× annual expenses
    annual_expenses_inr: float = 0.0,
) -> int | None:
    """
    Return the first year in which the MEDIAN portfolio value
    crosses the FI threshold (fire_multiple × annual expenses).
    Returns None if FI is never reached in the horizon.
    """
    target = fire_multiple * annual_expenses_inr
    if target <= 0:
        return None

    year_slices = extract_year_slices(paths, n_years)
    for t, values in enumerate(year_slices):
        med = percentile(values, 50)
        if med >= target:
            return t
    return None


def safe_withdrawal_rate(
    config:        PortfolioConfig,
    n_years:       int,
    n_simulations: int,
    target_success: float = 0.90,    # 90% success rate
    tolerance:     float = 0.001,
) -> float:
    """
    Binary search for the annual withdrawal rate (as fraction of initial portfolio)
    such that ≥ target_success fraction of simulations survive n_years.

    This is the Indian-market equivalent of the "4% rule".
    """
    lo, hi = 0.0, 0.30   # search between 0% and 30% withdrawal rate
    initial = config.total_value_inr

    for _ in range(20):   # 20 iterations → precision < 0.001%
        mid = (lo + hi) / 2
        annual_withdrawal = initial * mid

        # Build a test config with this withdrawal and no savings
        test_config = PortfolioConfig(
            total_value_inr      = initial,
            monthly_expenses_inr = annual_withdrawal / 12,
            monthly_savings_inr  = 0.0,
            assets               = config.assets,
        )
        test_paths    = run_simulation(test_config, n_years, n_simulations)
        ruin_probs    = ruin_probability_by_year(test_paths, n_years)
        success_rate  = 1.0 - ruin_probs[-1]

        if success_rate >= target_success:
            lo = mid    # can afford more
        else:
            hi = mid    # too aggressive

    return (lo + hi) / 2


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ASCII FAN CHART
# ═══════════════════════════════════════════════════════════════════════════════

# Percentile bands and their display characters
BANDS = [
    (90, "·", C_GREEN),
    (75, "░", C_GREEN),
    (50, "▒", C_YELLOW),
    (25, "▓", C_YELLOW),
    (10, "█", C_RED),
    (0,  "▓", C_RED),      # below 10th → same as 10th band
]

CHART_HEIGHT = 22     # rows (y-axis)
CHART_WIDTH  = 60     # columns (x-axis, one per year if possible)


def draw_fan_chart(
    year_slices:       list[list[float]],
    n_years:           int,
    fire_target:       float | None = None,
    ruin_probs:        list[float] | None = None,
    initial_value:     float = 0.0,
) -> None:
    """
    Draw an ASCII fan chart showing the distribution of wealth across time.

    Y-axis: wealth in ₹ Cr (Crore)
    X-axis: years
    Bands : 10th / 25th / 50th / 75th / 90th percentile ranges
    """
    # ── Compute percentile tracks ─────────────────────────────────────────
    pct_tracks: dict[int, list[float]] = {}
    for pct in [10, 25, 50, 75, 90]:
        pct_tracks[pct] = [percentile(year_slices[t], pct) for t in range(n_years + 1)]

    # ── Y-axis range ──────────────────────────────────────────────────────
    y_max = max(pct_tracks[90]) * 1.05
    y_max_cr = y_max / 1e7          # convert to Crore
    y_min_cr = 0.0

    # ── Step sizes ────────────────────────────────────────────────────────
    x_step = max(1, n_years // CHART_WIDTH)     # years per column
    x_cols = n_years // x_step + 1
    y_step = (y_max_cr - y_min_cr) / CHART_HEIGHT

    def y_to_row(value_inr: float) -> int:
        cr = value_inr / 1e7
        row = CHART_HEIGHT - int((cr - y_min_cr) / y_step)
        return max(0, min(CHART_HEIGHT, row))

    def row_to_cr(row: int) -> float:
        return y_max_cr - row * y_step

    # ── Build grid ────────────────────────────────────────────────────────
    # grid[row][col] = (char, colour)
    grid: list[list[tuple[str, str]]] = [
        [(" ", C_RESET) for _ in range(x_cols)]
        for _ in range(CHART_HEIGHT + 1)
    ]

    for col in range(x_cols):
        year = col * x_step
        if year > n_years:
            break

        p90 = pct_tracks[90][year]
        p75 = pct_tracks[75][year]
        p50 = pct_tracks[50][year]
        p25 = pct_tracks[25][year]
        p10 = pct_tracks[10][year]

        for row in range(CHART_HEIGHT + 1):
            val = row_to_cr(row) * 1e7

            if   val >= p90:  char, col_code = "·", C_GREEN
            elif val >= p75:  char, col_code = "░", C_GREEN
            elif val >= p50:  char, col_code = "▒", C_YELLOW
            elif val >= p25:  char, col_code = "▓", C_YELLOW
            elif val >= p10:  char, col_code = "▓", C_RED
            elif val > 0:     char, col_code = "█", C_RED
            else:             char, col_code = " ", C_RESET

            # Mark FIRE line
            if fire_target is not None:
                fire_row = y_to_row(fire_target)
                if row == fire_row:
                    char, col_code = "─", C_CYAN

            grid[row][col] = (char, col_code)

    # ── Print chart ───────────────────────────────────────────────────────
    title = "  WEALTH TRAJECTORY FAN CHART  (10,000 simulated futures)"
    print(f"\n{C_BOLD}{C_WHITE}{title}{C_RESET}")
    print(f"  {C_DIM}Each band = a range of possible outcomes. Wider = more uncertainty.{C_RESET}\n")

    for row in range(CHART_HEIGHT + 1):
        cr = row_to_cr(row)
        label = f"  ₹{cr:5.1f}Cr │"
        print(f"{C_DIM}{label}{C_RESET}", end="")
        for col in range(x_cols):
            ch, colour = grid[row][col]
            print(f"{colour}{ch}{C_RESET}", end="")
        print()

    # X-axis
    x_axis_label = "          └" + "─" * x_cols
    print(f"{C_DIM}{x_axis_label}{C_RESET}")
    year_labels = "           "
    for col in range(0, x_cols, max(1, x_cols // 8)):
        year = col * x_step
        year_labels += f"{year:<8}"
    print(f"{C_DIM}{year_labels}  Year{C_RESET}")

    # Legend
    print(f"\n  {C_GREEN}·{C_RESET} 90th pct  "
          f"{C_GREEN}░{C_RESET} 75th pct  "
          f"{C_YELLOW}▒{C_RESET} Median  "
          f"{C_YELLOW}▓{C_RESET} 25th pct  "
          f"{C_RED}█{C_RESET} 10th pct  "
          + (f"  {C_CYAN}─{C_RESET} FIRE target" if fire_target else ""))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. REPORT PRINTER
# ═══════════════════════════════════════════════════════════════════════════════

SEP  = "═" * 68
SEP2 = "─" * 68


def _inr(v: float) -> str:
    if v >= 1e7:
        return f"₹{v/1e7:.2f} Cr"
    if v >= 1e5:
        return f"₹{v/1e5:.2f} L"
    return f"₹{v:,.0f}"


def _pct(v: float) -> str:
    return f"{v*100:.1f}%"


def _risk_colour(prob: float) -> str:
    if prob < 0.05:  return C_GREEN
    if prob < 0.15:  return C_YELLOW
    return C_RED


def print_header(config: PortfolioConfig, n_years: int, n_simulations: int) -> None:
    print(f"\n{C_BOLD}{SEP}{C_RESET}")
    print(f"{C_BOLD}  MONTE CARLO WEALTH RUNWAY SIMULATOR{C_RESET}")
    print(f"  {C_DIM}{n_simulations:,} simulated futures · {n_years}-year horizon · Indian market parameters{C_RESET}")
    print(f"{C_BOLD}{SEP}{C_RESET}")

    print(f"\n  {C_BOLD}Portfolio Snapshot{C_RESET}")
    print(f"  {'Total value':<30} {_inr(config.total_value_inr)}")
    print(f"  {'Monthly expenses':<30} {_inr(config.monthly_expenses_inr)}")
    print(f"  {'Monthly savings':<30} {_inr(config.monthly_savings_inr)}")
    print(f"  {'Annual expense ratio':<30} {_pct(config.monthly_expenses_inr * 12 / config.total_value_inr)}")

    print(f"\n  {C_BOLD}Asset Allocation{C_RESET}")
    for a in config.assets:
        name   = a["name"]
        params = ASSET_PARAMS.get(name, ASSET_PARAMS["CASH"])
        print(
            f"  {name:<10} {a['allocation_pct']:>5.1f}%  "
            f"  {C_DIM}expected {_pct(params.mean_return)}/yr, "
            f"std {_pct(params.annual_std)}{C_RESET}"
        )
    print()


def print_outcome_table(
    year_slices:   list[list[float]],
    ruin_probs:    list[float],
    n_years:       int,
    checkpoints:   list[int] | None = None,
) -> None:
    if checkpoints is None:
        checkpoints = [1, 3, 5, 10, 15, 20, 25, 30]
    checkpoints = [y for y in checkpoints if y <= n_years]

    print(f"\n  {C_BOLD}Outcome Distribution by Year{C_RESET}")
    print(f"  {C_DIM}{'Year':<6} {'10th pct':>12} {'Median':>12} {'90th pct':>12} {'Ruin prob':>12}{C_RESET}")
    print(f"  {SEP2}")

    for yr in checkpoints:
        vals   = year_slices[yr]
        p10    = percentile(vals, 10)
        p50    = percentile(vals, 50)
        p90    = percentile(vals, 90)
        ruin   = ruin_probs[yr]
        rc     = _risk_colour(ruin)
        print(
            f"  {yr:<6} {_inr(p10):>12} {_inr(p50):>12} {_inr(p90):>12} "
            f"  {rc}{_pct(ruin):>9}{C_RESET}"
        )


def print_key_metrics(
    paths:         list[list[float]],
    ruin_probs:    list[float],
    n_years:       int,
    config:        PortfolioConfig,
    swr:           float,
    fire_year:     int | None,
) -> None:
    final_values = [p[-1] for p in paths]
    final_ruin   = ruin_probs[-1]
    median_final = percentile(final_values, 50)
    median_cagr  = statistics.median([cagr_from_path(p) for p in paths if p[-1] > 0])

    rc = _risk_colour(final_ruin)

    print(f"\n  {C_BOLD}Key Metrics{C_RESET}")
    print(f"  {SEP2}")
    print(f"  {'Probability of ruin ('}{n_years}yr):{'':<3} {rc}{_pct(final_ruin)}{C_RESET}")
    print(f"  {'Median portfolio in '}{n_years}yr:   {_inr(median_final)}")
    print(f"  {'Median real CAGR:':<30} {_pct(median_cagr)}")
    print(f"  {'Safe withdrawal rate (90% success):':<5} {_pct(swr)}/yr  "
          f"{C_DIM}(={_inr(config.total_value_inr * swr)}/yr){C_RESET}")

    if fire_year is not None:
        print(f"  {'Financial independence (25× rule)':<5} Year {fire_year}  "
              f"{C_DIM}(target: {_inr(config.monthly_expenses_inr * 12 * 25)}){C_RESET}")
    else:
        print(f"  {'Financial independence (25× rule)':<5} {C_RED}Not reached in {n_years}yr{C_RESET}")

    # Sequence of returns risk warning
    p10_5yr  = percentile([p[min(5, n_years)] for p in paths], 10)
    p50_5yr  = percentile([p[min(5, n_years)] for p in paths], 50)
    sorr_gap = (p50_5yr - p10_5yr) / max(p50_5yr, 1)
    print(f"\n  {C_BOLD}Sequence-of-Returns Risk (Year 5){C_RESET}")
    print(f"  {C_DIM}Gap between median and 10th percentile at year 5{C_RESET}")
    print(f"  Median: {_inr(p50_5yr)}  |  10th pct: {_inr(p10_5yr)}  "
          f"|  Gap: {_pct(sorr_gap)} downside")
    if sorr_gap > 0.40:
        print(f"  {C_RED}⚠  High sequence risk. A bad early sequence could be irreversible.{C_RESET}")
    elif sorr_gap > 0.20:
        print(f"  {C_YELLOW}⚡ Moderate sequence risk. Consider a cash buffer for year 1-3 expenses.{C_RESET}")
    else:
        print(f"  {C_GREEN}✔  Low sequence risk. Portfolio is relatively resilient to bad early years.{C_RESET}")


def print_verdict(ruin_prob: float, swr: float, fire_year: int | None, n_years: int) -> None:
    print(f"\n  {C_BOLD}Verdict{C_RESET}")
    print(f"  {SEP2}")

    if ruin_prob < 0.05:
        verdict = "STRONG"
        colour  = C_GREEN
        msg     = "Your portfolio has excellent survival odds. You are on track."
    elif ruin_prob < 0.15:
        verdict = "MODERATE"
        colour  = C_YELLOW
        msg     = "Survivable but not comfortable. Small improvements would significantly reduce risk."
    elif ruin_prob < 0.30:
        verdict = "FRAGILE"
        colour  = C_YELLOW
        msg     = "Real risk of outliving your money. Consider reducing expenses or increasing savings."
    else:
        verdict = "DANGER"
        colour  = C_RED
        msg     = "High probability of portfolio ruin. Structural changes needed urgently."

    print(f"  {colour}{C_BOLD}  {verdict}{C_RESET}  —  {msg}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  PRODUCT SUGGESTION NOTE
# ═══════════════════════════════════════════════════════════════════════════════

def print_product_suggestion() -> None:
    note = f"""
{C_BOLD}{SEP}{C_RESET}
{C_BOLD}  WHY THIS BELONGS IN TIMECELL.AI{C_RESET}
{SEP2}

  Most Indian wealth apps — and many globally — show users a single projected
  number. "At 12% CAGR, your ₹1 Cr becomes ₹9.6 Cr in 20 years." This is
  accurate on average and dangerously misleading in practice.

  The problem: {C_BOLD}sequence of returns risk{C_RESET}.

  If you retire and the first 3 years are bad (-30%, -20%, +5%), you are
  forced to sell assets at a loss to cover expenses. Those sold assets don't
  participate in the recovery. A portfolio that "should" have lasted 30 years
  runs dry in 12. The average return was fine. The sequence destroyed it.

  Monte Carlo simulation is the correct mental model. timecell.ai could
  differentiate itself by being the first Indian wealth platform that shows
  clients a {C_CYAN}probability cloud{C_RESET}, not a line — and teaches them what it means.

{C_BOLD}  Concrete product additions:{C_RESET}

  1. {C_YELLOW}Runway confidence band{C_RESET} — replace the single projection chart with
     a fan chart. Show 10th / 50th / 90th percentile. Let the user see
     how wide the cone of uncertainty is. Wide = risky; narrow = stable.

  2. {C_YELLOW}Safe Withdrawal Rate calculator{C_RESET} — "How much can I safely spend
     per year?" India's SWR is likely {C_BOLD}3-3.5%{C_RESET} (vs the US "4% rule")
     because of higher inflation and different return profiles. timecell
     should tell clients this number, not the American one.

  3. {C_YELLOW}FIRE year tracker{C_RESET} — "You will reach Financial Independence in
     {C_BOLD}Year 14{C_RESET} (median scenario)." Gamified, trackable, motivating.

  4. {C_YELLOW}Sequence-of-returns stress test{C_RESET} — "What if the first 3 years
     of your retirement are the worst 3 years in NIFTY history?" Show
     the client what happens. This is a genuinely sobering, useful tool.

  5. {C_YELLOW}Portfolio resilience score{C_RESET} — a single number (0-100) derived from
     the ruin probability, SWR, and SORR gap. Shows on the dashboard.
     "Your resilience score: 74/100. Here's what would raise it to 85."

{C_DIM}  The insight: people don't optimise for expected returns.
  They optimise to avoid catastrophe. Show them the catastrophe space.{C_RESET}

{SEP}
"""
    print(note)


# ═══════════════════════════════════════════════════════════════════════════════
# 8.  CLI & ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monte Carlo Wealth Runway Simulator — probabilistic financial planning."
    )
    parser.add_argument(
        "--mode",
        choices=["drawdown", "accumulation"],
        default="drawdown",
        help="'drawdown' = retired/spending mode; 'accumulation' = saving mode (default: drawdown)",
    )
    parser.add_argument(
        "--years", type=int, default=20,
        help="Simulation horizon in years (default: 20)",
    )
    parser.add_argument(
        "--sims", type=int, default=10_000,
        help="Number of Monte Carlo simulations (default: 10,000)",
    )
    parser.add_argument(
        "--no-swr", action="store_true",
        help="Skip safe withdrawal rate calculation (faster)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # ── Portfolio definition (same structure as Task 1 — fully swappable) ──
    if args.mode == "drawdown":
        # Retired investor: spending down the portfolio
        config = PortfolioConfig(
            total_value_inr      = 10_000_000,    # ₹1 Cr
            monthly_expenses_inr =     80_000,    # ₹80K/month
            monthly_savings_inr  =          0,    # no income
            assets=[
                {"name": "BTC",     "allocation_pct": 30},
                {"name": "NIFTY50", "allocation_pct": 40},
                {"name": "GOLD",    "allocation_pct": 20},
                {"name": "CASH",    "allocation_pct": 10},
            ],
        )
    else:
        # Accumulation investor: still working and saving
        config = PortfolioConfig(
            total_value_inr      =  5_000_000,    # ₹50L current portfolio
            monthly_expenses_inr =     50_000,    # ₹50K/month expenses
            monthly_savings_inr  =     30_000,    # ₹30K/month savings
            assets=[
                {"name": "NIFTY50", "allocation_pct": 60},
                {"name": "GOLD",    "allocation_pct": 20},
                {"name": "FD",      "allocation_pct": 10},
                {"name": "CASH",    "allocation_pct": 10},
            ],
        )

    n_years = args.years
    n_sims  = args.sims

    print_header(config, n_years, n_sims)

    # ── Run simulation ────────────────────────────────────────────────────
    print(f"  {C_DIM}Running {n_sims:,} simulations…{C_RESET}", end="", flush=True)
    paths = run_simulation(config, n_years, n_sims)
    print(f"\r  {C_GREEN}✔{C_RESET} {n_sims:,} simulations complete.          ")

    # ── Analytics ─────────────────────────────────────────────────────────
    year_slices = extract_year_slices(paths, n_years)
    ruin_probs  = ruin_probability_by_year(paths, n_years)

    annual_expenses = config.monthly_expenses_inr * 12
    fire_target = annual_expenses * 25 if annual_expenses > 0 else None
    fire_year   = find_fire_year(paths, n_years,
                                  fire_multiple=25.0,
                                  annual_expenses_inr=annual_expenses)

    # SWR is slow (runs another simulation loop); skip with --no-swr
    if args.no_swr or annual_expenses <= 0:
        swr = 0.0
    else:
        print(f"  {C_DIM}Computing safe withdrawal rate…{C_RESET}", end="", flush=True)
        swr = safe_withdrawal_rate(config, n_years, n_sims // 5)
        print(f"\r  {C_GREEN}✔{C_RESET} SWR computed.                         ")

    # ── Fan chart ─────────────────────────────────────────────────────────
    draw_fan_chart(year_slices, n_years, fire_target=fire_target, ruin_probs=ruin_probs,
                   initial_value=config.total_value_inr)

    # ── Tables & metrics ──────────────────────────────────────────────────
    checkpoints = [1, 3, 5, 10, 15, 20, 25, 30]
    print_outcome_table(year_slices, ruin_probs, n_years, checkpoints)
    print_key_metrics(paths, ruin_probs, n_years, config, swr, fire_year)
    print_verdict(ruin_probs[-1], swr, fire_year, n_years)

    # ── Product suggestion ────────────────────────────────────────────────
    print_product_suggestion()


if __name__ == "__main__":
    main()
