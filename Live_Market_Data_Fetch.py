"""
Live Market Data Fetcher
Fetches current prices for BTC, NIFTY50, and GOLD from free public APIs.
Prints a clean formatted table with graceful error handling.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Optional

# ---------------------------------------------------------------------------
# Logging setup — INFO to stdout, errors stand out cleanly
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AssetPrice:
    name: str
    symbol: str
    price: Optional[float]
    currency: str
    source: str
    error: Optional[str] = None

    @property
    def fetched_ok(self) -> bool:
        return self.price is not None and self.error is None


# ---------------------------------------------------------------------------
# Fetchers — each returns an AssetPrice; never raises
# ---------------------------------------------------------------------------

def _fetch_crypto_coingecko(coin_id: str, display_name: str, vs_currency: str = "usd") -> AssetPrice:
    """
    Fetch crypto price from CoinGecko public API (no key required).
    https://api.coingecko.com/api/v3/simple/price
    """
    import urllib.request, json, urllib.error

    url = (
        f"https://api.coingecko.com/api/v3/simple/price"
        f"?ids={coin_id}&vs_currencies={vs_currency}"
    )
    asset = AssetPrice(
        name=display_name,
        symbol=coin_id.upper(),
        price=None,
        currency=vs_currency.upper(),
        source="CoinGecko",
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        asset.price = float(data[coin_id][vs_currency])
        logger.info("Fetched %s via CoinGecko: %s %.2f", display_name, vs_currency.upper(), asset.price)
    except urllib.error.HTTPError as exc:
        asset.error = f"HTTP {exc.code}: {exc.reason}"
        logger.error("CoinGecko fetch failed for %s — %s", display_name, asset.error)
    except Exception as exc:
        asset.error = str(exc)
        logger.error("CoinGecko fetch failed for %s — %s", display_name, asset.error)
    return asset


def _fetch_yfinance(ticker: str, display_name: str, currency: str) -> AssetPrice:
    """
    Fetch stock / index / commodity price via yfinance (free, no key).
    """
    asset = AssetPrice(
        name=display_name,
        symbol=ticker,
        price=None,
        currency=currency,
        source="Yahoo Finance",
    )
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).fast_info
        price = info.last_price
        if price is None or price != price:          # NaN guard
            raise ValueError("Received null/NaN price from Yahoo Finance.")
        asset.price = float(price)
        logger.info("Fetched %s via Yahoo Finance: %s %.2f", display_name, currency, asset.price)
    except Exception as exc:
        asset.error = str(exc)
        logger.error("Yahoo Finance fetch failed for %s — %s", display_name, asset.error)
    return asset


# ---------------------------------------------------------------------------
# Asset definitions & orchestration
# ---------------------------------------------------------------------------

def fetch_all_prices() -> list[AssetPrice]:
    """
    Fetch prices for BTC (CoinGecko), NIFTY50 (Yahoo Finance), GOLD (Yahoo Finance).
    Each fetcher is independent — a failure in one doesn't affect the others.
    """
    assets: list[AssetPrice] = []

    # Crypto — CoinGecko
    assets.append(_fetch_crypto_coingecko("bitcoin", "Bitcoin (BTC)", vs_currency="usd"))

    # Indian index — Yahoo Finance (^NSEI = NIFTY 50)
    assets.append(_fetch_yfinance("^NSEI", "NIFTY 50", currency="INR"))

    # Gold ETF in INR — Yahoo Finance (GOLDBEES.NS = Nippon India Gold BeES)
    assets.append(_fetch_yfinance("GC=F", "Gold (COMEX)", currency="USD"))

    return assets


# ---------------------------------------------------------------------------
# Table renderer
# ---------------------------------------------------------------------------

COL_ASSET    = 20
COL_PRICE    = 16
COL_CURRENCY = 10
COL_SOURCE   = 16


def _separator(char: str = "─") -> str:
    total = COL_ASSET + COL_PRICE + COL_CURRENCY + COL_SOURCE + 13
    return char * total


def _fmt_price(price: Optional[float], error: Optional[str]) -> str:
    if price is not None:
        return f"{price:,.2f}"
    return f"ERROR: {(error or 'unknown')[:20]}"


def print_price_table(assets: list[AssetPrice]) -> None:
    """Render a clean terminal table of asset prices."""
    IST = timezone(timedelta(hours=5, minutes=30))
    timestamp = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")

    print()
    print(f"  Asset Prices — fetched at {timestamp}")
    print(_separator("═"))
    print(
        f"  {'Asset':<{COL_ASSET}} "
        f"{'Price':>{COL_PRICE}} "
        f"{'Currency':<{COL_CURRENCY}} "
        f"{'Source':<{COL_SOURCE}}"
    )
    print(_separator("─"))

    for ap in assets:
        price_str = _fmt_price(ap.price, ap.error)
        status_indicator = "✔" if ap.fetched_ok else "✘"
        print(
            f"  {status_indicator} {ap.name:<{COL_ASSET - 2}} "
            f"{price_str:>{COL_PRICE}} "
            f"{ap.currency:<{COL_CURRENCY}} "
            f"{ap.source:<{COL_SOURCE}}"
        )

    print(_separator("═"))
    ok_count  = sum(1 for a in assets if a.fetched_ok)
    fail_count = len(assets) - ok_count
    print(f"  {ok_count}/{len(assets)} fetched successfully", end="")
    if fail_count:
        print(f"  |  {fail_count} failed (see errors above)", end="")
    print("\n")


# ---------------------------------------------------------------------------
# Demo mode — used when live APIs are unreachable (e.g. sandboxed environment)
# ---------------------------------------------------------------------------

def demo_prices() -> list[AssetPrice]:
    """Return realistic hardcoded prices for demonstration / offline testing."""
    from datetime import date
    note = "(demo — live API blocked in this environment)"
    return [
        AssetPrice("Bitcoin (BTC)",  "BTC",    62_341.20, "USD", f"CoinGecko {note}"),
        AssetPrice("NIFTY 50",       "^NSEI",  22_541.80, "INR", f"Yahoo Finance {note}"),
        AssetPrice("Gold (COMEX)",   "GC=F",    2_320.50, "USD", f"Yahoo Finance {note}"),
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prices = fetch_all_prices()

    # If all fetches failed (e.g. network sandbox), fall back to demo data
    if not any(a.fetched_ok for a in prices):
        logger.info(
            "All live fetches failed — displaying demo data. "
            "Run outside a sandboxed environment for real prices."
        )
        prices = demo_prices()

    print_price_table(prices)
