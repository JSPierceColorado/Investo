#!/usr/bin/env python3
"""
IBKR Super Dooper Bot — VIG + XRP + WSB (S&P 500 filtered)
==========================================================
Runs perpetually, cycling every 15 minutes. Implements three strategies:

1) VIG investor
   - If RSI(14) <= 30 AND SMA(60) of 15‑min bars < SMA(240) of 15‑min bars,
     buy 20% of available USD funds of VIG at market. Never sells VIG.

2) XRP investor
   - If RSI(14) <= 30 AND SMA(60) < SMA(240) on 15‑min bars,
     buy 10% of available USD funds of XRP at market.

3) WallStreetBets (WSB) scoop
   - Scrape tickers from https://www.reddit.com/r/wallstreetbets/
   - **Filter to S&P 500 constituents only** (no sector/denylist logic per your latest request)
   - For each qualifying ticker: if RSI(14) <= 30 AND SMA(60) < SMA(240) on 15‑min bars,
     buy 5% of available USD funds at market.

Before any buys on each cycle, first process existing holdings:
   - For each non‑VIG position: if RSI(14) >= 70 AND SMA(60) > SMA(240) on 15‑min bars
     AND the position has >= +5% profit, SELL ALL of that asset at market.
   - VIG is NEVER sold by this bot.

Connection: Interactive Brokers via `ib_insync`; requires TWS/IBG up and API enabled.

Environment Variables
---------------------
IB_HOST=127.0.0.1
IB_PORT=7497               # 7497 paper, 7496 live (typical)
IB_CLIENT_ID=42            # unique int per client
IB_ACCOUNT=U1234567        # optional; will pick the first managed account if omitted
ENABLE_XRP=true            # set false to disable XRP module if your account can’t trade it
MAX_WSB_POSTS=50           # how many subreddit posts to scan each cycle

Python Requirements
-------------------
- ib_insync>=0.9.84
- pandas>=2.0
- numpy>=1.24
- requests>=2.31
- lxml, html5lib, beautifulsoup4  (for pandas.read_html)

Run:
  python super_dooper_ibkr_bot.py

Notes
-----
- Test in PAPER first. Understand risks.
- Assumes fractional shares OK for ETFs; adjust rounding if not.
- XRP via IBKR Crypto (Paxos). Set ENABLE_XRP=false if unsupported.
- Uses 15‑min historical bars; enough history pulled for SMA(240).
- Stocks/ETFs orders are DAY, crypto GTC; change as needed.
"""

from __future__ import annotations
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import List, Set, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import requests

from ib_insync import (
    IB,
    util,
    Stock,
    Crypto,
    Contract,
    MarketOrder,
    ContractDetails,
)

# ----------------------------- Configuration ----------------------------- #
VIG_SYMBOL = "VIG"            # Vanguard Dividend Appreciation ETF
VIG_EXCHANGE = "ARCA"

XRP_SYMBOL = "XRP"
XRP_EXCHANGE = "PAXOS"        # IBKR Crypto via Paxos
ENABLE_XRP = os.getenv('ENABLE_XRP', 'true').lower() in {'1','true','yes','y'}

HISTORY_DAYS = 15
BAR_SIZE = '15 mins'
DURATION_STR = f"{HISTORY_DAYS} D"
WHAT_TO_SHOW_STOCK = 'TRADES'
WHAT_TO_SHOW_CRYPTO = 'TRADES'

ALLOC_VIG = 0.20              # 20% of available USD funds
ALLOC_XRP = 0.10              # 10% of available USD funds
ALLOC_WSB = 0.05              # 5% of available USD funds per qualifying ticker

MIN_ORDER_USD = 1.00          # ignore tiny orders
ALLOW_OUTSIDE_RTH = False     # stock/ETF orders outside regular hours

WSB_JSON_URL = "https://www.reddit.com/r/wallstreetbets/.json"
WSB_HEADERS = {"User-Agent": "IBKR-SDB/1.2 (by u/super_dooper)"}
TICKER_REGEX = re.compile(r"[A-Z]{1,5}")
STOP_TICKERS = {
    'A','I','YOLO','DD','WSB','TOS','OTM','ITM','ATH','FD','IV','ETF','CEO','CFO','CTO',
    'SEC','USA','AM','PM','GDP','EPS','FOMO','IMO','TLDR','CPI','PPI','FUD','ATH','PDT'
}
MAX_WSB_POSTS = int(os.getenv('MAX_WSB_POSTS', '50'))

CYCLE_MINUTES = 15

# ----------------------- S&P 500 constituents ----------------------------- #
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def get_sp500_tickers() -> Set[str]:
    try:
        # use pandas.read_html; requires lxml/html5lib/bs4 installed
        tables = pd.read_html(requests.get(SP500_URL, timeout=20).text)
        df = tables[0]
        sym_col = next((c for c in df.columns if str(c).lower().startswith('symbol')), 'Symbol')
        tickers = set(str(t).upper().replace('.', '-') for t in df[sym_col].tolist())
        print(f"[SP500] Loaded {len(tickers)} symbols")
        return tickers
    except Exception as e:
        print(f"[SP500][WARN] Could not fetch S&P 500 list: {e}")
        return set()

SP500_TICKERS = get_sp500_tickers()

# ---------------------------- Utility Functions --------------------------- #

def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def sleep_until_next_quarter(minute_granularity: int = CYCLE_MINUTES):
    now = datetime.now()
    minutes = (now.minute // minute_granularity + 1) * minute_granularity
    next_tick = now.replace(second=0, microsecond=0)
    if minutes >= 60:
        next_tick = next_tick.replace(minute=0) + timedelta(hours=1)
    else:
        next_tick = next_tick.replace(minute=minutes)
    delay = max(5, (next_tick - now).total_seconds())
    print(f"[Timing] Sleeping {delay:.1f}s until {next_tick} local...")
    time.sleep(delay)

# ---------------------------- Indicator Math ------------------------------ #

def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi_val = 100.0 - (100.0 / (1.0 + rs))
    return rsi_val.fillna(50.0)

# ------------------------------ IB Helpers -------------------------------- #

def connect_ib() -> IB:
    host = os.getenv('IB_HOST', '127.0.0.1')
    port = int(os.getenv('IB_PORT', '7497'))
    client_id = int(os.getenv('IB_CLIENT_ID', '42'))

    ib = IB()
    print(f"[IB] Connecting to {host}:{port} clientId={client_id} ...")
    ib.connect(host, port, clientId=client_id, readonly=False, timeout=10)
    if not ib.isConnected():
        raise RuntimeError("Could not connect to IB. Ensure TWS/IB Gateway is running and API is enabled.")

    # pick an account if not provided
    acct = os.getenv('IB_ACCOUNT')
    accounts = ib.managedAccounts() or []
    if not acct:
        if accounts:
            acct = accounts[0]
            print(f"[IB] Using first managed account: {acct}")
        else:
            print("[IB] No managed accounts found; proceeding without explicit account.")
    else:
        if accounts and acct not in accounts:
            print(f"[IB][WARN] Provided IB_ACCOUNT={acct} not in managed accounts {accounts}. Continuing anyway.")

    ib.wrapper.account = acct
    return ib


def get_available_usd_funds(ib: IB) -> float:
    try:
        summary = ib.accountSummary()
    except Exception as e:
        print(f"[IB][WARN] accountSummary failed: {e}")
        return 0.0

    usd = 0.0
    acct = ib.wrapper.account
    for row in summary:
        if row.tag in ('AvailableFunds', 'TotalCashValue') and row.currency == 'USD':
            if acct and hasattr(row, 'account') and row.account and row.account != acct:
                continue
            try:
                val = float(row.value)
                if row.tag == 'AvailableFunds':
                    usd = max(usd, val)
                elif row.tag == 'TotalCashValue' and usd == 0.0:
                    usd = val
            except ValueError:
                pass
    print(f"[Funds] Available USD funds: ${usd:,.2f}")
    return max(usd, 0.0)


def resolve_contract(ib: IB, raw: Contract) -> Contract:
    cd: List[ContractDetails] = ib.reqContractDetails(raw)
    if not cd:
        raise RuntimeError(f"Unable to resolve contract: {raw}")
    return cd[0].contract


def vig_contract(ib: IB) -> Contract:
    c = Stock(VIG_SYMBOL, 'SMART', 'USD', primaryExchange=VIG_EXCHANGE)
    try:
        return resolve_contract(ib, c)
    except Exception:
        return resolve_contract(ib, Stock(VIG_SYMBOL, VIG_EXCHANGE, 'USD'))


def xrp_contract(ib: IB) -> Contract:
    return resolve_contract(ib, Crypto(XRP_SYMBOL, XRP_EXCHANGE, 'USD'))


def generic_stock_contract(ib: IB, symbol: str) -> Optional[Contract]:
    try:
        return resolve_contract(ib, Stock(symbol, 'SMART', 'USD'))
    except Exception:
        for ex in ('ARCA', 'NASDAQ', 'NYSE'):
            try:
                return resolve_contract(ib, Stock(symbol, ex, 'USD'))
            except Exception:
                continue
    return None


def fetch_15min_bars(ib: IB, contract: Contract) -> pd.DataFrame:
    what = WHAT_TO_SHOW_CRYPTO if contract.secType == 'CRYPTO' else WHAT_TO_SHOW_STOCK
    bars = ib.reqHistoricalData(
        contract,
        endDateTime='',
        durationStr=DURATION_STR,
        barSizeSetting=BAR_SIZE,
        whatToShow=what,
        useRTH=False,
        formatDate=1,
        keepUpToDate=False,
    )
    if not bars:
        raise RuntimeError(f"No historical bars for {contract.conId or contract.symbol}")
    df = util.df(bars)
    if 'close' not in df.columns:
        raise RuntimeError(f"Historical data missing 'close' for {contract.symbol}")
    return df


def last_indicators(df: pd.DataFrame) -> Dict[str, float]:
    close = df['close']
    return {
        'rsi14': float(rsi(close, 14).iloc[-1]),
        'sma60': float(sma(close, 60).iloc[-1]),
        'sma240': float(sma(close, 240).iloc[-1]),
        'last': float(close.iloc[-1]),
    }


def place_market_value_order(ib: IB, contract: Contract, usd_amount: float, side: str) -> Optional[int]:
    if usd_amount < MIN_ORDER_USD:
        print(f"[Order] Skipping {side} for {contract.symbol}: amount ${usd_amount:.2f} < ${MIN_ORDER_USD:.2f}")
        return None

    df = fetch_15min_bars(ib, contract)
    last = float(df['close'].iloc[-1])
    if last <= 0:
        print(f"[Order] Invalid last price for {contract.symbol}")
        return None

    qty = usd_amount / last

    if contract.secType == 'STK':
        qty = round(qty, 3)  # fractional to 0.001
    elif contract.secType == 'CRYPTO':
        qty = round(qty, 6)
    else:
        qty = max(1, int(qty))

    if qty <= 0:
        print(f"[Order] Non‑positive quantity for {contract.symbol}")
        return None

    action = 'BUY' if side.upper() == 'BUY' else 'SELL'
    order = MarketOrder(action, qty, tif='DAY' if contract.secType == 'STK' else 'GTC', outsideRth=ALLOW_OUTSIDE_RTH)
    print(f"[Order] {action} {qty} {contract.symbol} ~${usd_amount:.2f} @ ~{last:.2f}")
    trade = ib.placeOrder(contract, order)
    ib.sleep(1)
    return trade.order.orderId if trade and trade.order else None

# --------------------------- Strategy Conditions -------------------------- #

def buy_condition(ind: Dict[str, float]) -> bool:
    return (ind['rsi14'] <= 30.0) and (ind['sma60'] < ind['sma240'])


def sell_condition(ind: Dict[str, float]) -> bool:
    return (ind['rsi14'] >= 70.0) and (ind['sma60'] > ind['sma240'])

# ----------------------------- WSB Scraper -------------------------------- #

def extract_tickers_from_text(text: str) -> Set[str]:
    symbols = set(TICKER_REGEX.findall(text or ''))
    filtered = set()
    for s in symbols:
        if s in STOP_TICKERS:
            continue
        if len(s) == 1 and s not in {'F','T'}:
            continue
        filtered.add(s)
    return filtered


def fetch_wsb_tickers(limit_posts: int = MAX_WSB_POSTS) -> List[str]:
    try:
        resp = requests.get(WSB_JSON_URL, headers=WSB_HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"[WSB][WARN] Failed to fetch subreddit JSON: {e}")
        return []

    tickers: Set[str] = set()
    children = data.get('data', {}).get('children', [])
    for i, item in enumerate(children):
        if i >= limit_posts:
            break
        post = item.get('data', {})
        title = post.get('title', '')
        selftext = post.get('selftext', '')
        tickers |= extract_tickers_from_text(title)
        tickers |= extract_tickers_from_text(selftext)

    # filter to S&P 500 only
    tickers &= SP500_TICKERS

    out = sorted(tickers)
    if out:
        print(f"[WSB] Found S&P500 tickers: {', '.join(out)}")
    else:
        print("[WSB] No valid S&P500 tickers found.")
    return out

# ------------------------------ Core Logic -------------------------------- #

def process_positions_for_sales(ib: IB):
    print("[Sell-Check] Evaluating existing positions for exits…")
    portfolio = ib.portfolio()
    for pos in portfolio:
        contract = pos.contract
        symbol = contract.symbol
        sec_type = contract.secType

        if sec_type == 'STK' and symbol.upper() == VIG_SYMBOL:
            print("[Sell-Check] Skipping VIG (never sold)")
            continue

        try:
            df = fetch_15min_bars(ib, contract)
            ind = last_indicators(df)
        except Exception as e:
            print(f"[Sell-Check][WARN] Could not fetch indicators for {symbol}: {e}")
            continue

        current_price = ind['last']
        avg_cost = float(pos.avgCost or 0.0)
        if avg_cost <= 0:
            profit_ok = False
            print(f"[Sell-Check][WARN] Avg cost missing for {symbol}")
        else:
            profit_pct = (current_price - avg_cost) / avg_cost
            profit_ok = (profit_pct >= 0.05)

        cond = sell_condition(ind)
        print(f"[Sell-Check] {symbol} rsi={ind['rsi14']:.1f} sma60={ind['sma60']:.2f} sma240={ind['sma240']:.2f} last={current_price:.2f} profitOK={profit_ok}")

        if cond and profit_ok:
            qty = pos.position
            if qty <= 0:
                print(f"[Sell-Check] {symbol} not a long position; skipping per spec")
                continue
            usd_amount = abs(qty) * current_price
            try:
                place_market_value_order(ib, contract, usd_amount, 'SELL')
                print(f"[Sell-Check] Submitted SELL ALL for {symbol} (qty={qty})")
            except Exception as e:
                print(f"[Sell-Check][ERR] Failed to submit sell for {symbol}: {e}")


def run_vig_investor(ib: IB):
    try:
        c = vig_contract(ib)
        df = fetch_15min_bars(ib, c)
        ind = last_indicators(df)
        print(f"[VIG] rsi={ind['rsi14']:.1f} sma60={ind['sma60']:.2f} sma240={ind['sma240']:.2f} last={ind['last']:.2f}")
        if buy_condition(ind):
            funds = get_available_usd_funds(ib)
            usd = funds * ALLOC_VIG
            place_market_value_order(ib, c, usd, 'BUY')
        else:
            print("[VIG] No buy signal this cycle.")
    except Exception as e:
        print(f"[VIG][ERR] {e}")


def run_xrp_investor(ib: IB):
    if not ENABLE_XRP:
        print("[XRP] Disabled by config.")
        return
    try:
        c = xrp_contract(ib)
        df = fetch_15min_bars(ib, c)
        ind = last_indicators(df)
        print(f"[XRP] rsi={ind['rsi14']:.1f} sma60={ind['sma60']:.2f} sma240={ind['sma240']:.2f} last={ind['last']:.6f}")
        if buy_condition(ind):
            funds = get_available_usd_funds(ib)
            usd = funds * ALLOC_XRP
            place_market_value_order(ib, c, usd, 'BUY')
        else:
            print("[XRP] No buy signal this cycle.")
    except Exception as e:
        print(f"[XRP][ERR] {e}")


def run_wsb_investor(ib: IB):
    tickers = fetch_wsb_tickers()
    if not tickers:
        return

    for sym in tickers:
        c = generic_stock_contract(ib, sym)
        if not c:
            print(f"[WSB] Skipping {sym}: could not resolve contract")
            continue
        try:
            df = fetch_15min_bars(ib, c)
            ind = last_indicators(df)
            print(f"[WSB] {sym} rsi={ind['rsi14']:.1f} sma60={ind['sma60']:.2f} sma240={ind['sma240']:.2f} last={ind['last']:.2f}")
            if buy_condition(ind):
                funds = get_available_usd_funds(ib)
                usd = funds * ALLOC_WSB
                place_market_value_order(ib, c, usd, 'BUY')
            else:
                print(f"[WSB] {sym}: No buy signal.")
        except Exception as e:
            print(f"[WSB][WARN] {sym}: {e}")

# ------------------------------ Main Runner ------------------------------- #

def run_cycle(ib: IB):
    print("================= CYCLE START =================")
    print(f"[Time] {datetime.now().isoformat(timespec='seconds')}")

    process_positions_for_sales(ib)
    run_vig_investor(ib)
    run_xrp_investor(ib)
    run_wsb_investor(ib)

    print("================== CYCLE END ==================
")


def main():
    ib = connect_ib()
    try:
        while True:
            try:
                run_cycle(ib)
            except Exception as e:
                print(f"[Cycle][ERR] {e}")
            sleep_until_next_quarter(CYCLE_MINUTES)
    finally:
        if ib.isConnected():
            ib.disconnect()


if __name__ == '__main__':
    main()
