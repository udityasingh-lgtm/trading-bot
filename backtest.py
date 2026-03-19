"""
============================================================
  Backtester — Test the AI model on historical data
  Usage: python backtest.py
============================================================
"""

import pandas as pd
import numpy as np
import logging
from bot import get_client, fetch_ohlcv, add_features, create_labels, train_model, load_model, get_signal, FEATURE_COLS
from config import CONFIG
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def backtest(symbol="BTCUSDT", interval="1h", initial_balance=1000.0,
             risk_pct=0.10, stop_loss=2.0, take_profit=4.0,
             confidence=0.62, fee=0.001):

    log.info("📈 Starting Backtest...")
    client = get_client()

    df = fetch_ohlcv(client, symbol, interval, limit=1000)
    df = add_features(df)

    # Train on first 70%, test on last 30%
    split = int(len(df) * 0.70)
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()

    model_path = "backtest_model.pkl"
    model, scaler = train_model(train_df.copy(), model_path)

    # ── Simulate Trading ──────────────────────────────────
    balance     = initial_balance
    in_position = False
    entry_price = 0
    entry_qty   = 0
    trades      = []
    equity_curve = []

    for i in range(len(test_df)):
        row_df = test_df.iloc[:i+1]
        if len(row_df) < 2:
            continue

        price = row_df["close"].iloc[-1]
        equity_curve.append({"time": row_df.index[-1], "equity": balance})

        # Check SL/TP
        if in_position:
            change = (price - entry_price) / entry_price * 100
            if change <= -stop_loss or change >= take_profit:
                reason = "SL" if change <= -stop_loss else "TP"
                sell_value = entry_qty * price * (1 - fee)
                pnl = sell_value - (entry_qty * entry_price)
                balance += sell_value
                trades.append({
                    "type": "SELL", "reason": reason,
                    "price": price, "qty": entry_qty,
                    "pnl_usd": round(pnl, 2), "pnl_pct": round(change, 2),
                    "balance": round(balance, 2)
                })
                in_position = False
                continue

        # Get signal (only every few candles to simulate real-time)
        if i % 1 == 0:
            try:
                latest = row_df[FEATURE_COLS].iloc[-1:]
                latest_scaled = scaler.transform(latest)
                prob = model.predict_proba(latest_scaled)[0]
                buy_prob  = prob[1]
                sell_prob = prob[0]

                if buy_prob >= confidence and not in_position and balance > 10:
                    invest = balance * risk_pct
                    entry_qty   = (invest / price) * (1 - fee)
                    entry_price = price
                    balance    -= invest
                    in_position = True
                    trades.append({
                        "type": "BUY", "reason": "AI",
                        "price": price, "qty": round(entry_qty, 6),
                        "pnl_usd": 0, "pnl_pct": 0,
                        "balance": round(balance, 2)
                    })

                elif sell_prob >= confidence and in_position:
                    sell_value = entry_qty * price * (1 - fee)
                    change_pct = (price - entry_price) / entry_price * 100
                    pnl = sell_value - (entry_qty * entry_price)
                    balance += sell_value
                    trades.append({
                        "type": "SELL", "reason": "AI",
                        "price": price, "qty": round(entry_qty, 6),
                        "pnl_usd": round(pnl, 2), "pnl_pct": round(change_pct, 2),
                        "balance": round(balance, 2)
                    })
                    in_position = False

            except Exception:
                pass

    # ── Results ───────────────────────────────────────────
    trades_df = pd.DataFrame(trades)
    sells = trades_df[trades_df["type"] == "SELL"] if len(trades_df) > 0 else pd.DataFrame()

    total_return = (balance - initial_balance) / initial_balance * 100
    n_trades     = len(sells)
    win_trades   = len(sells[sells["pnl_usd"] > 0]) if n_trades > 0 else 0
    win_rate     = win_trades / n_trades * 100 if n_trades > 0 else 0
    total_pnl    = sells["pnl_usd"].sum() if n_trades > 0 else 0

    print("\n" + "="*50)
    print("         BACKTEST RESULTS")
    print("="*50)
    print(f"  Symbol        : {symbol} ({interval})")
    print(f"  Test Period   : {test_df.index[0].date()} → {test_df.index[-1].date()}")
    print(f"  Starting $    : ${initial_balance:,.2f}")
    print(f"  Final $       : ${balance:,.2f}")
    print(f"  Total Return  : {total_return:+.2f}%")
    print(f"  Total PnL     : ${total_pnl:+.2f}")
    print(f"  Total Trades  : {n_trades}")
    print(f"  Win Rate      : {win_rate:.1f}%")
    print("="*50 + "\n")

    trades_df.to_csv("backtest_trades.csv", index=False)
    pd.DataFrame(equity_curve).to_csv("equity_curve.csv", index=False)
    log.info("📊 Results saved to backtest_trades.csv and equity_curve.csv")

    return balance, trades_df


if __name__ == "__main__":
    backtest(
        symbol=CONFIG["symbol"],
        interval=CONFIG["interval"],
        risk_pct=CONFIG["risk_pct"],
        stop_loss=CONFIG["stop_loss_pct"],
        take_profit=CONFIG["take_profit_pct"],
        confidence=CONFIG["confidence_threshold"]
    )
