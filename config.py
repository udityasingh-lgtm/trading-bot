"""
============================================================
  Bot Configuration v2
  Technical + News + Fear & Greed
============================================================
"""

CONFIG = {
    # ── API Keys ──────────────────────────────────────────
    "api_key":    "DoTY2EYdHLuYr0dCvwJO4NPFsu0vICwnKxCF4kgWyINmyWmHzdAovSVulA3xpLq5",   # Leave empty - Railway uses environment variables
    "api_secret": "WYWyG7KpFHvC0HbDytyq7UooqvInvbQoZQcvZKJ3G46xyAkdLypbVURizUf1Xo4C",   # Leave empty - Railway uses environment variables

    # ── Multiple Trading Pairs ────────────────────────────
    "symbols": [
        "BTCUSDT",    # Solana  - very active, lots of signals
    ],

    # ── Candle Interval ───────────────────────────────────
    "interval": "5m",   # 5 minutes - good balance of speed and accuracy

    # ── Safety ────────────────────────────────────────────
    "testnet": True,    # True = demo account, False = real money

    # ── Risk Management ───────────────────────────────────
    "risk_pct":           0.90,   # Use 90% of per-coin budget
    "stop_loss_pct":      3.0,    # Stop loss at -3%
    "take_profit_pct":    5.0,    # Take profit at +5%
    "max_daily_loss_pct": 10.0,   # Stop if down 10% today
    "max_trades_per_day": 10,     # Max 10 trades per day
    "min_usdt_trade":     10.0,   # Minimum $10 per trade

    # ── AI Settings ───────────────────────────────────────
    "confidence_threshold": 0.10,  # Technical signal threshold

    # ── Timing ────────────────────────────────────────────
    "sleep_seconds":          30,   # Check every 2 minutes
    "retrain_every_n_cycles": 50,    # Retrain AI every 50 cycles
}

"""
Signal Weights:
  Technical Analysis : 50%
  Fear & Greed Index : 30%
  News Sentiment     : 20%

BUY when combined score >= 0.25
SELL when combined score <= -0.25
"""
