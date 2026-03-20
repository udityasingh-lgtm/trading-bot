CONFIG = {
    # API Keys - leave empty, Railway uses environment variables
    "api_key":    "",
    "api_secret": "",
    "testnet": True,

    # Trading
    "symbols":  ["SOLUSDT"],
    "interval": "1m",

    # Risk
    "risk_pct":           0.95,
    "stop_loss_pct":      0.3,
    "take_profit_pct":    0.5,
    "min_usdt_trade":     5.0,
    "max_daily_loss_pct": 5.0,
    "max_trades_per_day": 50,

    # AI
    "confidence_threshold":   0.52,   # Lower = more trades
    "retrain_every_n_cycles": 30,     # Retrain every 30 cycles

    # Timing
    "sleep_seconds": 30,
}
