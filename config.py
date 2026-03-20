"""
============================================================
  Bot Configuration
  Edit this file to customize your trading bot
============================================================
"""

CONFIG = {
    # ── API Keys ──────────────────────────────────────────
    # Better: set these as environment variables instead
    # export BINANCE_API_KEY="your_key"
    # export BINANCE_API_SECRET="your_secret"
    "api_key":    "DoTY2EYdHLuYr0dCvwJO4NPFsu0vICwnKxCF4kgWyINmyWmHzdAovSVulA3xpLq5",
    "api_secret": "WYWyG7KpFHvC0HbDytyq7UooqvInvbQoZQcvZKJ3G46xyAkdLypbVURizUf1Xo4C",

    # ── Trading Pair & Interval ───────────────────────────
    "symbol":   "SOLUSDT",     # Pair to trade
    "interval": "1h",          # Candle interval: 1m, 5m, 15m, 1h, 4h, 1d

    # ── Safety: Use Testnet First! ────────────────────────
    "testnet": True,           # ⚠️ Set to False only for real trading

    # ── Risk Management ───────────────────────────────────
    "risk_pct":          0.10,   # % of USDT balance to use per trade (10%)
    "stop_loss_pct":     2.0,    # Stop-loss at -2%
    "take_profit_pct":   4.0,    # Take-profit at +4%
    "max_daily_loss_pct": 5.0,   # Stop bot if daily PnL drops -5%
    "max_trades_per_day": 5,     # Max number of trades per day
    "min_usdt_trade":    10.0,   # Minimum USDT needed to place a trade

    # ── AI Signal Settings ────────────────────────────────
    "confidence_threshold": 0.62,  # Min model confidence to act (62%)

    # ── Timing ────────────────────────────────────────────
    "sleep_seconds": 120,         # Seconds between each bot cycle (1h = 3600)
    "retrain_every_n_cycles": 24,  # Retrain AI every N cycles
}

"""
───── Interval Reference ──────────────────────────────
  "1m"   → 1 minute   (very aggressive, high fees)
  "5m"   → 5 minutes  (aggressive)
  "15m"  → 15 minutes (moderate)
  "1h"   → 1 hour     (recommended for beginners)
  "4h"   → 4 hours    (safer, fewer trades)
  "1d"   → 1 day      (swing trading)

───── Recommended Starting Settings ──────────────────
  - interval: "1h"
  - risk_pct: 0.05 to 0.10 (5–10%)
  - stop_loss_pct: 2.0
  - take_profit_pct: 3.0 to 5.0
  - confidence_threshold: 0.60 to 0.65
  - testnet: True (ALWAYS start here!)
"""
