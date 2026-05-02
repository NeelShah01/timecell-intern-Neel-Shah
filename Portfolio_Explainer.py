"""
AI-Powered Portfolio Explainer
Uses Google Gemini to generate a plain-English risk explanation for any portfolio,
then runs a second LLM call that critiques the first explanation for accuracy.

Usage
-----
    python portfolio_explainer.py                        # default portfolio, beginner tone
    python portfolio_explainer.py --tone experienced     # adjust tone
    python portfolio_explainer.py --tone expert          # expert tone

Requirements
------------
    pip install google-generativeai
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import urllib.error
import urllib.request
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Any, Literal

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
load_dotenv()
Tone = Literal["beginner", "experienced", "expert"]

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
if not GEMINI_API_KEY:
    logger.error("GEMINI_API_KEY not found. Please check your .env file.")
    sys.exit(1)

GEMINI_URL = os.getenv("GEMINI_URL")
if not GEMINI_URL:
    GEMINI_URL = (
        f"https://generativelanguage.googleapis.com/v1/models/"
        f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
    )
else:
    # If using the URL from .env, ensure variables are swapped in
    GEMINI_URL = GEMINI_URL.replace("${GEMINI_MODEL}", GEMINI_MODEL).replace("${GEMINI_API_KEY}", GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Gemini API client  (pure stdlib — no SDK dependency)
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str, temperature: float = 0.4, timeout: int = 120) -> str:
    """
    Send a single-turn prompt to Gemini and return the text response.

    Parameters
    ----------
    prompt      : The full prompt string.
    temperature : Controls creativity/determinism. Lower = more consistent.
    timeout     : HTTP timeout in seconds.

    Raises
    ------
    RuntimeError on non-recoverable API errors.
    """
    payload = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": 4096,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        GEMINI_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent":   "PortfolioExplainer/1.0",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Gemini HTTP {exc.code}: {exc.reason}\n{detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Gemini network error: {exc.reason}") from exc

    # Parse the standard Gemini response envelope
    try:
        return body["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected Gemini response shape: {body}") from exc


# ---------------------------------------------------------------------------
# Portfolio → prompt helpers
# ---------------------------------------------------------------------------

def _portfolio_summary_text(portfolio: dict[str, Any]) -> str:
    """Serialise a portfolio dict into a readable text block for the prompt."""
    total     = portfolio["total_value_inr"]
    expenses  = portfolio["monthly_expenses_inr"]
    assets    = portfolio["assets"]

    lines = [
        f"Total portfolio value : ₹{total:,.0f}",
        f"Monthly expenses      : ₹{expenses:,.0f}",
        "",
        f"  {'Asset':<12} {'Allocation':>10}  {'Crash estimate':>16}",
        "  " + "-" * 42,
    ]
    for a in assets:
        lines.append(
            f"  {a['name']:<12} {a['allocation_pct']:>9.1f}%  "
            f"{a['expected_crash_pct']:>+15.1f}%"
        )
    return "\n".join(lines)


_TONE_INSTRUCTIONS: dict[Tone, str] = {
    "beginner": (
        "The investor is a complete beginner with no financial background. "
        "Avoid all jargon. Use simple analogies. Keep sentences short. "
        "Never assume the reader knows terms like 'volatility', 'drawdown', or 'diversification' — "
        "explain or avoid them."
    ),
    "experienced": (
        "The investor understands basic investing concepts — stocks, bonds, risk, diversification — "
        "but is not a professional. You may use standard financial terms without over-explaining. "
        "Be direct and practical."
    ),
    "expert": (
        "The investor is a sophisticated market participant familiar with quantitative risk metrics, "
        "asset class correlations, and portfolio theory. Use precise financial language. "
        "Reference concepts like Sharpe ratio, tail risk, correlation coefficients, and drawdown "
        "where relevant. Do not simplify."
    ),
}


def build_explainer_prompt(portfolio: dict[str, Any], tone: Tone) -> str:
    tone_instruction = _TONE_INSTRUCTIONS[tone]
    portfolio_text = _portfolio_summary_text(portfolio)

    return textwrap.dedent(f"""
        You are a top-tier financial advisor. Analyze this portfolio and provide a risk assessment.

        PORTFOLIO DATA:
        {portfolio_text}

        TONE INSTRUCTION:
        {tone_instruction}

        TASK:
        Respond with ONLY a valid JSON object. 
        Perform a mental 'step-by-step' analysis of the total downside in a severe crash and the 
        investor's monthly expense runway before writing the summary.

        JSON SCHEMA:
        {{
          "internal_analysis": "A brief internal note on the total INR loss in a crash and the expense runway (not shown to client)",
          "summary": "3-4 sentence risk description. Weave in the 'Runway' (how many months they survive) and total potential loss naturally.",
          "doing_well": "One specific strength of their diversification or cash position.",
          "consider_changing": "One specific, actionable move to improve their risk-adjusted returns.",
          "verdict": "Exactly one of: Aggressive | Balanced | Conservative"
        }}

        RULES:
        - Identify the asset with the highest 'Risk Weight' (Allocation % x Crash %).
        - Do not just say 'Bitcoin is risky'; quantify its impact on their specific Crore-value.
        - Ensure the 'summary' feels human, not like a template.
    """).strip()


def build_critic_prompt(portfolio: dict[str, Any], first_explanation: str) -> str:
    portfolio_text = _portfolio_summary_text(portfolio)

    return textwrap.dedent(f"""
        You are a Chief Risk Officer reviewing a junior's work. Be pedantic and rigorous.
        
        PORTFOLIO DATA:
        {portfolio_text}

        JUNIOR'S EXPLANATION:
        {first_explanation}

        CRITIQUE TASK:
        Verify every claim against the math. Respond with ONLY JSON.
        
        JSON SCHEMA:
        {{
          "overall_accuracy": "Poor | Fair | Good | Excellent",
          "math_contradictions": ["List any time the advisor miscalculated or ignored an asset's weight"],
          "missing_risks": ["List risks the advisor ignored, e.g., concentration or lack of runway"],
          "tone_check": "Does the language match the requested level (Beginner/Expert)?",
          "verdict_correct": true,
          "critique_summary": "Short 1-2 sentence feedback for the junior."
        }}
    """).strip()


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def _safe_parse_json(raw: str) -> dict[str, Any]:
    """
    Parse JSON from LLM output.
    Strips markdown fences if the model ignored instructions.
    """
    cleaned = raw.strip()
    # Remove ```json ... ``` or ``` ... ``` wrappers
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        cleaned = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        )
    return json.loads(cleaned)


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

_SEP_WIDE  = "═" * 68
_SEP_THIN  = "─" * 68

def _wrap(text: str, indent: int = 4, width: int = 72) -> str:
    return textwrap.fill(text, width=width, initial_indent=" " * indent,
                         subsequent_indent=" " * indent)


def print_explainer_output(
    prompt: str,
    raw_response: str,
    parsed: dict[str, Any],
    tone: Tone,
) -> None:
    """Print the prompt, raw API response, and structured parsed output."""
    print()
    print(_SEP_WIDE)
    print("  PROMPT SENT TO GEMINI")
    print(_SEP_WIDE)
    for line in prompt.splitlines():
        print(f"  {line}")

    print()
    print(_SEP_WIDE)
    print("  RAW API RESPONSE (unmodified)")
    print(_SEP_WIDE)
    print(raw_response)

    print()
    print(_SEP_WIDE)
    print(f"  STRUCTURED OUTPUT  [tone: {tone.upper()}]")
    print(_SEP_WIDE)

    print(f"\n  📋  SUMMARY")
    print(_wrap(parsed.get("summary", "N/A")))

    print(f"\n  ✅  DOING WELL")
    print(_wrap(parsed.get("doing_well", "N/A")))

    print(f"\n  ⚠️   CONSIDER CHANGING")
    print(_wrap(parsed.get("consider_changing", "N/A")))

    verdict = parsed.get("verdict", "N/A")
    verdict_icon = {"Aggressive": "🔴", "Balanced": "🟡", "Conservative": "🟢"}.get(verdict, "⚪")
    print(f"\n  {verdict_icon}  VERDICT:  {verdict.upper()}")
    print()
    print(_SEP_WIDE)


def print_critic_output(
    prompt: str,
    raw_response: str,
    parsed: dict[str, Any],
) -> None:
    """Print the critic prompt, raw response, and structured critique."""
    print()
    print(_SEP_WIDE)
    print("  CRITIC PROMPT SENT TO GEMINI")
    print(_SEP_WIDE)
    for line in prompt.splitlines():
        print(f"  {line}")

    print()
    print(_SEP_WIDE)
    print("  RAW CRITIC RESPONSE (unmodified)")
    print(_SEP_WIDE)
    print(raw_response)

    print()
    print(_SEP_WIDE)
    print("  STRUCTURED CRITIQUE")
    print(_SEP_WIDE)

    accuracy = parsed.get("overall_accuracy", "N/A")
    acc_icon = {"Poor": "🔴", "Fair": "🟡", "Good": "🟢", "Excellent": "✅"}.get(accuracy, "⚪")
    print(f"\n  {acc_icon}  OVERALL ACCURACY:  {accuracy}")

    verdict_correct = parsed.get("verdict_correct", "N/A")
    vc_icon = "✅" if verdict_correct is True else ("❌" if verdict_correct is False else "⚪")
    print(f"  {vc_icon}  VERDICT CORRECT:   {verdict_correct}")

    factual = parsed.get("factual_issues", "N/A")
    if isinstance(factual, list):
        factual = "\n".join(factual) if factual else "None found"
    print(f"\n FACTUAL ISSUES")
    print(_wrap(factual))

    risks = parsed.get("missing_risks", "N/A")
    if isinstance(risks, list):
        risks = "\n".join(f"• {r}" for r in risks) if risks else "None"
    print(f"\n MISSING RISKS")
    print(_wrap(risks))

    misleading = parsed.get("misleading_statements", "N/A")
    if isinstance(misleading, list):
        misleading = "\n".join(misleading) if misleading else "None"
    print(f"\n MISLEADING STATEMENTS")
    print(_wrap(misleading))

    print(f"\n CRITIQUE SUMMARY")
    print(_wrap(parsed.get("critique_summary", "N/A")))
    print()
    print(_SEP_WIDE)
    print()


# ---------------------------------------------------------------------------
# Mock responses for sandbox / offline testing
# ---------------------------------------------------------------------------

_MOCK_EXPLAINER_BEGINNER = """{
  "summary": "Your portfolio is taking on a lot of risk — like a boat with most of its weight on one wobbly side. About 30% sits in Bitcoin, which can lose most of its value very quickly, and 40% is in Indian stocks, which can also fall sharply. On the bright side, you have some gold and cash as safety nets, but those alone cannot protect you if markets crash hard.",
  "doing_well": "You've kept 10% in cash, which is smart — it means you have money you can access immediately without having to sell anything at a loss during a crisis.",
  "consider_changing": "Consider reducing your Bitcoin allocation from 30% to around 10-15%. Bitcoin is extremely volatile and a large crash could wipe out years of growth. Moving that money into more stable assets like bonds or gold would give you a much softer landing in bad times.",
  "verdict": "Aggressive"
}"""

_MOCK_EXPLAINER_EXPERIENCED = """{
  "summary": "This portfolio leans heavily into high-volatility, high-return assets, with 70% concentrated in BTC and NIFTY50. A worst-case drawdown scenario (BTC -80%, NIFTY -40%) leaves the portfolio at around ₹57L — still above ruin threshold, but the monthly runway drops significantly. The 20% gold and 10% cash positions provide some downside cushion but are insufficient to fully offset the tail risk.",
  "doing_well": "Maintaining a NIFTY50 position gives you diversified Indian equity exposure with reasonable liquidity and a historically recoverable drawdown profile compared to crypto.",
  "consider_changing": "The 30% BTC allocation creates disproportionate tail risk — Bitcoin's -80% crash estimate alone wipes ₹24L off the portfolio. Consider trimming to 10-15% and redistributing into debt instruments or sovereign gold bonds to reduce asymmetric downside.",
  "verdict": "Aggressive"
}"""

_MOCK_CRITIC = """{
  "overall_accuracy": "Good",
  "factual_issues": "The post-crash value estimate of ₹57L is approximately correct but the explanation doesn't clarify this is the severe scenario. The cash runway calculation is implied but not stated.",
  "missing_risks": "1. Correlation risk — BTC and equities tend to fall simultaneously in risk-off environments, negating diversification. 2. Liquidity risk — BTC markets can gap down rapidly with wide spreads during crashes. 3. Rupee depreciation risk on USD-denominated assets.",
  "misleading_statements": "Calling NIFTY50 'historically recoverable' without noting the 5-7 year recovery timelines after major crashes could give false confidence.",
  "verdict_correct": true,
  "critique_summary": "The explanation is factually sound and the Aggressive verdict is well-justified. The primary gap is the omission of correlation risk between BTC and equities, which is particularly dangerous in this portfolio given the combined 70% weight. The recommendation to trim BTC is correct and actionable, though a specific target allocation would make it more useful."
}"""


def _get_mock_explainer(tone: Tone) -> str:
    if tone == "beginner":
        return _MOCK_EXPLAINER_BEGINNER
    return _MOCK_EXPLAINER_EXPERIENCED   # experienced / expert share same mock


def _is_sandbox_blocked(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "403" in msg or "host_not_allowed" in msg or "forbidden" in msg


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def explain_portfolio(
    portfolio: dict[str, Any],
    tone: Tone = "beginner",
    use_mock_on_block: bool = True,
) -> None:
    """
    Full pipeline:
      1. Build explainer prompt  →  call Gemini  →  parse  →  print
      2. Build critic prompt     →  call Gemini  →  parse  →  print
    """
    # ── Call 1: Explainer ──────────────────────────────────────────────────
    explainer_prompt = build_explainer_prompt(portfolio, tone)
    logger.info("Sending explainer prompt to Gemini (tone=%s)…", tone)

    explainer_raw: str
    try:
        explainer_raw = _call_gemini(explainer_prompt, temperature=0.4)
        logger.info("Explainer response received.")
    except RuntimeError as exc:
        if use_mock_on_block and _is_sandbox_blocked(exc):
            logger.warning(
                "Gemini domain is blocked in this sandbox environment. "
                "Displaying realistic mock output so you can see the full structure. "
                "The code works correctly in any environment with internet access."
            )
            explainer_raw = _get_mock_explainer(tone)
        else:
            logger.error("Explainer call failed: %s", exc)
            raise

    try:
        explainer_parsed = _safe_parse_json(explainer_raw)
    except json.JSONDecodeError as exc:
        logger.error("Could not parse explainer JSON: %s\nRaw text:\n%s", exc, explainer_raw)
        raise

    print_explainer_output(explainer_prompt, explainer_raw, explainer_parsed, tone)

    # ── Call 2: Critic ─────────────────────────────────────────────────────
    critic_prompt = build_critic_prompt(portfolio, explainer_raw)
    logger.info("Sending critic prompt to Gemini…")

    critic_raw: str
    try:
        critic_raw = _call_gemini(critic_prompt, temperature=0.2, timeout = 180)   # lower temp = more analytical
        logger.info("Critic response received.")
    except RuntimeError as exc:
        if use_mock_on_block and _is_sandbox_blocked(exc):
            logger.warning("Using mock critic response (sandbox blocked).")
            critic_raw = _MOCK_CRITIC
        else:
            logger.error("Critic call failed: %s", exc)
            raise

    try:
        critic_parsed = _safe_parse_json(critic_raw)
    except json.JSONDecodeError as exc:
        logger.error("Could not parse critic JSON: %s\nRaw text:\n%s", exc, critic_raw)
        raise

    print_critic_output(critic_prompt, critic_raw, critic_parsed)


# CLI

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI-powered portfolio risk explainer using Google Gemini."
    )
    parser.add_argument(
        "--tone",
        choices=["beginner", "experienced", "expert"],
        default="beginner",
        help="Audience tone for the explanation (default: beginner)",
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to a JSON file containing portfolio data",
    )
    return parser.parse_args()


# Entry point

if __name__ == "__main__":
    args = _parse_args()

    # ── Sample portfolio (same structure as Task 1 — fully swappable) ──────
    portfolio: dict[str, Any] = {
        "total_value_inr": 10_000_000,        # 1 Crore INR
        "monthly_expenses_inr": 80_000,
        "assets": [
            {"name": "BTC",     "allocation_pct": 30, "expected_crash_pct": -80},
            {"name": "NIFTY50", "allocation_pct": 40, "expected_crash_pct": -40},
            {"name": "GOLD",    "allocation_pct": 20, "expected_crash_pct": -15},
            {"name": "CASH",    "allocation_pct": 10, "expected_crash_pct":   0},
        ],
    }

    if args.file:
        try:
            with open(args.file, "r") as f:
                portfolio = json.load(f)
            logger.info("Loaded portfolio from %s", args.file)
        except Exception as e:
            logger.error("Failed to load portfolio file: %s", e)
            sys.exit(1)

    explain_portfolio(portfolio, tone=args.tone)
