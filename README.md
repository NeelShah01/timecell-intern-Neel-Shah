# timecell-intern-Neel-Shah
### TimeCell.ai · Summer Internship 2026 · Technical Assessment

---

## Quick Start

```bash
pip install yfinance

python portfolio_risk.py        # Task 1
python market_data.py           # Task 2
python portfolio_explainer.py --tone beginner    # Task 3
python portfolio_explainer.py --tone experienced
python portfolio_explainer.py --tone expert
python monte_carlo.py           # Task 4
python monte_carlo.py --mode accumulation --years 30
```

**Python:** 3.10+ · **External deps:** `yfinance` (Task 2 only) · Everything else is standard library.

---

## Task 01 — Portfolio Risk Calculator
## Task 01 — Portfolio Risk Calculator

**File:** `portfolio_risk.py`

### What it computes

`compute_risk_metrics(portfolio)` returns a dictionary with all five required fields:

| Field | How it's computed |
|---|---|
| `post_crash_value` | Σ (asset_value × (1 + crash_pct / 100)) across all assets |
| `runway_months` | post_crash_value ÷ monthly_expenses |
| `ruin_test` | `'PASS'` if runway > 12 months, `'FAIL'` otherwise |
| `largest_risk_asset` | `max(allocation_pct × |crash_pct|)` — a weighted score, not just the largest allocation |
| `concentration_warning` | `True` if any single asset > 40% of portfolio |

The function is called twice — once with `crash_multiplier=1.0` (severe) and once with `crash_multiplier=0.5` (moderate) — to produce both scenarios from the same logic. Both print side by side.

### Design decisions

**`largest_risk_asset` is a weighted score.** BTC at 30% allocation with -80% crash magnitude scores `2400`. NIFTY at 40% with -40% scores `1600`. The metric correctly flags BTC as the primary risk driver even though NIFTY has the larger allocation. Using raw allocation percentage alone would give the wrong answer.

**Validation runs before any math.** `_validate_portfolio()` raises a `ValueError` if allocations don't sum to 100%, rather than silently computing a wrong post-crash value.

**Zero-expense and 100%-cash edge cases are explicitly covered.** `runway_months` returns `float("inf")` when monthly expenses are zero. Both edge cases are demoed at the bottom of the script.

Moderate crash scenario (50% of expected crash magnitude) shown side by side with severe CLI bar chart using Unicode block characters, no external plotting libraries.

---

## Task 02 — Live Market Data Fetch
## Task 02 — Live Market Data Fetch

**File:** `market_data.py`

### Assets fetched

| Asset | Source | Currency |
|---|---|---|
| Bitcoin (BTC) | CoinGecko `/v3/simple/price` | USD |
| NIFTY 50 (`^NSEI`) | Yahoo Finance via `yfinance` | INR |
| Gold COMEX (`GC=F`) | Yahoo Finance via `yfinance` | USD |

All APIs are free with no key required.

### Error handling

Each asset has its own fetcher function (`_fetch_crypto_coingecko`, `_fetch_yfinance`) that catches all exceptions internally and returns an `AssetPrice` dataclass with the `error` field set. The orchestrator (`fetch_all_prices`) collects all three results regardless of what any individual fetcher does — one API going down doesn't stop the others from running.

Failures are logged with `[ERROR]` prefix and the table still renders, marking failed rows with `✘` and showing a truncated error message inline.

A few specifics:

- IST timestamp is constructed explicitly as `timezone(timedelta(hours=5, minutes=30))` — not system locale, which breaks on UTC servers.
- `yfinance` uses `fast_info.last_price` rather than downloading historical data. Much faster.
- A NaN guard (`price != price`) handles the case where Yahoo returns a float NaN for illiquid instruments rather than raising an error.
- If all three fetches fail (e.g. a sandboxed environment), `demo_prices()` activates automatically with a clear note that it's demo data.


---

## Task 03 — AI-Powered Portfolio Explainer
## Task 03 — AI-Powered Portfolio Explainer

**File:** `portfolio_explainer.py`

### What it does

Calls Gemini twice for any portfolio dictionary:

1. **Explainer call** — generates a structured plain-English risk explanation
2. **Critic call** — independently critiques the first explanation for factual accuracy and missing risks

Both the raw API response and the parsed structured output are printed separately, as required.

### Dynamic Data Input & Interaction
The script is not limited to just a hardcoded example but supports dynamic data ingestion:

* **Interactive Mode**: If run without flags, the script enters an interactive menu, prompting you to choose between the hardcoded 1 Crore INR example or a custom JSON file.
* **CLI File Ingestion**: Supports direct ingestion of any portfolio via the `--file` flag, bypassing the interactive menu for automated workflows:
  `python portfolio_explainer.py --file "path/to/your_portfolio.json"`
* **Path Flexibility**: Handles both relative and absolute file paths, including support for directory names containing spaces when wrapped in quotes.
* **Pre-flight Validation**: Automatically validates the JSON structure (checking for required keys like `total_value_inr` and `assets`) before initiating the Gemini API call.

### LLM used

**Google Gemini 2.5 Flash** via the REST API (`/v1beta/models/gemini-2.5-flash:generateContent`).

Chosen over GPT-4o and Claude for this task because Gemini Flash has a generous free tier with no waitlist, making the script runnable by anyone reviewing it without needing to manage credits. The prompt engineering matters more than the provider — the same approach works on any model.

Transport is pure stdlib `urllib`. No SDK dependency.

---

### Prompt Engineering — What I Tried, What Worked, What Changed

#### First attempt: open-ended prose

The initial prompt asked the model to respond as a "friendly financial advisor" and write a paragraph covering risk, strengths, weaknesses, and a verdict. The prose was natural but structurally inconsistent — the verdict sometimes appeared mid-sentence, sometimes used phrasing like "Moderately Aggressive" that didn't match the three required options. Parsing it reliably meant writing fragile regex, which isn't the right answer.

**Root cause:** Asking for structured content inside free-form prose means the model honours the spirit of the instruction but not the letter.

#### What changed: explicit JSON output contract

The prompt was rewritten with a clear output contract at the top:

```
Respond with ONLY a valid JSON object — no markdown, no code fences, no extra text.
```

The schema was spelled out with exact key names:

```json
{
  "summary": "3-4 sentence risk description",
  "doing_well": "one specific strength",
  "consider_changing": "one actionable improvement with reason",
  "verdict": "<exactly one of: Aggressive | Balanced | Conservative>"
}
```

The phrase "exactly one of" followed by the three options — with "nothing else" appended — eliminated ambiguous middle-ground answers entirely. The model stopped generating "Moderately Aggressive."

`_safe_parse_json()` strips markdown fences as a defensive fallback before calling `json.loads()`. This proved necessary: even with explicit instructions, the model occasionally wrapped JSON in triple backticks.

#### Tone system

Rather than interpolating a single word into the prompt, each tone level gets its own full instruction paragraph:

- **Beginner:** Avoid all jargon. Use analogies. Never assume the reader knows "volatility" or "drawdown" — define or avoid them.
- **Experienced:** Standard financial vocabulary is fine. Be direct, skip definitions.
- **Expert:** Use precise quantitative language — Sharpe ratio, tail risk, drawdown, correlation coefficients where relevant.

This produces meaningfully different outputs, not just a register shift. The beginner output used the phrase "like a boat with most of its weight on one wobbly side." The expert output referenced "asymmetric downside" and "tail risk concentration." A single interpolated word like `tone=beginner` wouldn't have produced that difference.

#### Temperature choices

- **Explainer: `0.4`** — enough creativity for natural prose, low enough to keep the data grounded and the JSON structure intact
- **Critic: `0.2`** — analytical tasks benefit from lower temperature; the critic needs to find specific factual errors, not be creative

#### The critic prompt

The second call receives the first call's **raw JSON output**, not the parsed version. The critic sees exactly what the model produced, including any formatting quirks. The framing is adversarial: "rigorous risk analyst reviewing a junior advisor's work" — not "review your own output." That framing produces more specific, less forgiving critiques.

The critic's output schema includes `verdict_correct: true|false` — a boolean check that the explainer's risk classification is consistent with the actual portfolio data. Most two-call LLM pipelines leave that loop open.

---

## Task 04 — The Open Problem
## Task 04 — The Open Problem

**File:** `monte_carlo.py`

### What I built: Monte Carlo Wealth Runway Simulator

```bash
python monte_carlo.py                        # drawdown mode, 20yr, 10,000 sims
python monte_carlo.py --mode accumulation    # saving/building wealth phase
python monte_carlo.py --years 30 --sims 5000
python monte_carlo.py --no-swr               # skip safe withdrawal rate calc
```

### Why this

Every wealth management tool I looked at — including most Indian ones — shows a single projected number. "At 12% CAGR, your ₹1 Cr becomes ₹9.6 Cr in 20 years." That's accurate on average. It's misleading in practice, because it ignores **sequence of returns risk**.

If you retire and the first three years are bad, you're forced to sell assets at depressed prices to cover living expenses. Those sold units don't participate in the recovery. A portfolio that "should" last 30 years can run dry in 12 — even if the average annual return looked fine on paper. Monte Carlo captures this by simulating 10,000 different orderings of returns, not just the average.

This is a gap that exists in Indian wealth tools specifically, and it's directly relevant to what TimeCell is building. The output includes a metric I'd advocate adding to TimeCell's portfolio temperature system: the **Sequence-of-Returns Risk score** — the gap between median and 10th percentile wealth at year 5. Crash percentage alone doesn't capture this.

### What it outputs

- **ASCII fan chart** — 10th / 25th / 50th / 75th / 90th percentile wealth paths across time, colour-coded with ANSI escape codes. No matplotlib.
- **Ruin probability table** — % chance of hitting ₹0 at years 1, 3, 5, 10, 15, 20
- **Safe Withdrawal Rate** — computed empirically via binary search, not assumed at 4%. For Indian investors, it comes out around 3–3.5% due to higher inflation. Most tools use the American number.
- **FIRE year** — the first year the median portfolio crosses 25× annual expenses
- **Sequence-of-Returns Risk score** — gap between median and 10th percentile at year 5, with a plain-English warning if it's high

### Key technical decisions

**Log-normal returns, not normal.** Normal distributions can generate returns below -100%, which is physically impossible. Log-normal prevents that and correctly captures the right-skewed nature of equity returns.

**Stochastic inflation.** Inflation is drawn each year from N(6%, 2%) based on Indian CPI. Expenses compound against a variable rate, not a fixed assumption.

**INR/USD depreciation.** BTC returns are in USD. Converted back to INR each year using a stochastic depreciation rate (~3.5%/yr). This reflects the actual experience of Indian investors holding dollar assets.

**Two modes** — `drawdown` for retired investors spending from the portfolio, `accumulation` for those still saving.

---

### AI Collaboration & Acknowledgments
These tasks were done using an AI-Augmented Engineering workflow. It helped in accelerating development and refining the user experience:
These tasks were done using an AI-Augmented Engineering workflow. It helped in accelerating development and refining the user experience:

Anthropic Claude: Assisted with logic refactoring, structural organization of the Monte Carlo simulation (Task 4), and drafting high-fidelity documentation.

Google Gemini: Served as both the primary runtime engine for the Portfolio Explainer (Task 3) and a development collaborator for debugging REST API payloads and optimizing CLI visualizations.

GitHub Copilot: Used throughout the development process for real-time code completion, boilerplate generation (specifically for argparse and logging setups), and drafting unit test structures.

## Hardest Part

The trickiest part of Task 3 was getting consistent structured output from the LLM across different portfolio inputs. The first prompt approach worked fine on the example portfolio but broke on edge cases, i.e., a 100%-cash portfolio would generate a verdict of "Conservative (approaching 0 risk)" which failed the string match. The fix was being more prescriptive about the output contract and adding the JSON fallback parser, but the lesson was that prompts need to be tested on boring edge cases, not just the interesting ones.

---