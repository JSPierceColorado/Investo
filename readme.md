# IBKR Super Dooper Bot — Railway Deploy Guide

## Overview
Runs on a 15-minute cycle, connecting to Interactive Brokers via API.
- **Sell-first**: closes profitable positions per rules (except VIG).
- **VIG**: buy-only, 20% allocation on signal.
- **XRP**: buy-only, 10% allocation (toggleable).
- **WSB**: scrapes r/wallstreetbets, filters to S&P 500 tickers, buys 5% allocation per ticker on signal.

## Networking note
Railway runs in the cloud. Your TWS/IB Gateway must be reachable from Railway:
- Run IB Gateway on a cloud VM and expose API with firewall + trusted IPs.
- Or tunnel/VPN.
- Or run both bot + IB Gateway on the same VM.

## Files
- `super_dooper_ibkr_bot.py`
- `requirements.txt`
- `Dockerfile`
- `.dockerignore`

## Environment Variables
Set in Railway → Variables:
- `IB_HOST` — host/IP of your IB Gateway
- `IB_PORT` — 7497 (paper) or 7496 (live)
- `IB_CLIENT_ID` — e.g., 42
- `IB_ACCOUNT` — optional, your IBKR account ID
- `ENABLE_XRP` — true/false (default true)
- `MAX_WSB_POSTS` — number of posts to scan (default 50)

## Build & Deploy on Railway
1. Push these files to GitHub.
2. Connect the repo in Railway (Deploy via Dockerfile).
3. Set env vars.
4. Deploy and monitor logs.

## IBKR Setup Checklist
- TWS or IB Gateway running.
- API enabled (“Enable ActiveX and Socket Clients”).
- Add Railway’s egress IPs to Trusted IPs.
- Trading permissions for Stocks/ETFs and Crypto (Paxos) if using XRP.

## Customization
- Adjust rounding in orders if no fractional shares.
- Change `ALLOW_OUTSIDE_RTH` for extended hours.
- Edit `MIN_ORDER_USD` for minimum order size.

## Local Test
```bash
export IB_HOST=127.0.0.1
export IB_PORT=7497
export IB_CLIENT_ID=42
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python super_dooper_ibkr_bot.py
