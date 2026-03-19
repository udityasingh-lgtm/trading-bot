"""
============================================================
  AI Trading Bot for Binance
  Author: Generated for you
  Usage: python bot.py
============================================================
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from binance.client import Client
from binance.exceptions import BinanceAPIException
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings("ignore")

from config import CONFIG

# ─── Logging Setup ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ─── Binance Client ───────────────────────────────────────
def get_client():
    api_key = os.getenv("BINANCE_API_KEY", CONFIG["api_key"])
    api_secret = os.getenv("BINANCE_API_SECRET", CONFIG["api_secret"])

    if CONFIG["testnet"]:
        client = Client(api_key, api_secret, testnet=True)
        client.API_URL = "https://testnet.binance.vision/api"
        log.info("🧪 Connected to Binance TESTNET")
    else:
        client = Client(api_key, api_secret)
        log.info("🚀 Connected to Binance LIVE")

    return client


# ─── Data Fetching ────────────────────────────────────────
def fetch_ohlcv(client, symbol, interval, limit=600):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "qav", "trades", "tbbav", "tbqav", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


# ─── Feature Engineering ──────────────────────────────────
def add_features(df):
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # Trend
    df["ema_9"]  = ta.trend.EMAIndicator(c, 9).ema_indicator()
    df["ema_21"] = ta.trend.EMAIndicator(c, 21).ema_indicator()
    df["ema_50"] = ta.trend.EMAIndicator(c, 50).ema_indicator()
    macd = ta.trend.MACD(c)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()
    df["adx"]         = ta.trend.ADXIndicator(h, l, c).adx()

    # Momentum
    df["rsi"]  = ta.momentum.RSIIndicator(c, 14).rsi()
    df["stoch_k"] = ta.momentum.StochasticOscillator(h, l, c).stoch()
    df["stoch_d"] = ta.momentum.StochasticOscillator(h, l, c).stoch_signal()
    df["cci"]  = ta.trend.CCIIndicator(h, l, c).cci()
    df["roc"]  = ta.momentum.ROCIndicator(c, 10).roc()

    # Volatility
    bb = ta.volatility.BollingerBands(c)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = bb.bollinger_wband()
    df["bb_pct"]   = bb.bollinger_pband()
    df["atr"]      = ta.volatility.AverageTrueRange(h, l, c).average_true_range()

    # Volume
    df["obv"]  = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["cmf"]  = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v).chaikin_money_flow()
    df["vwap"] = ta.volume.VolumeWeightedAveragePrice(h, l, c, v).volume_weighted_average_price()

    # Price-derived
    df["returns_1"] = c.pct_change(1)
    df["returns_3"] = c.pct_change(3)
    df["returns_5"] = c.pct_change(5)
    df["hl_ratio"]  = (h - l) / c
    df["oc_ratio"]  = (c - df["open"]) / df["open"]

    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "ema_9", "ema_21", "ema_50",
    "macd", "macd_signal", "macd_diff", "adx",
    "rsi", "stoch_k", "stoch_d", "cci", "roc",
    "bb_upper", "bb_lower", "bb_width", "bb_pct", "atr",
    "obv", "cmf", "vwap",
    "returns_1", "returns_3", "returns_5", "hl_ratio", "oc_ratio"
]


# ─── Label Creation ───────────────────────────────────────
def create_labels(df, lookahead=3, threshold=0.005):
    """
    1 = BUY  (price rises > threshold in next N candles)
    0 = HOLD/SELL
    """
    future_return = df["close"].shift(-lookahead) / df["close"] - 1
    df["target"] = (future_return > threshold).astype(int)
    return df.dropna()


# ─── Model Training ───────────────────────────────────────
def train_model(df, save_path="model.pkl"):
    log.info("📚 Training AI model...")
    df = create_labels(df)

    X = df[FEATURE_COLS]
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=20,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    log.info(f"✅ Model accuracy: {acc:.2%}")

    joblib.dump({"model": model, "scaler": scaler}, save_path)
    log.info(f"💾 Model saved to {save_path}")
    return model, scaler


def load_model(path="model.pkl"):
    data = joblib.load(path)
    return data["model"], data["scaler"]


# ─── Account & Risk ───────────────────────────────────────
def get_usdt_balance(client):
    balance = client.get_asset_balance(asset="USDT")
    return float(balance["free"])


def get_asset_balance(client, asset):
    bal = client.get_asset_balance(asset=asset)
    return float(bal["free"])


def calculate_quantity(client, symbol, usdt_amount):
    info = client.get_symbol_info(symbol)
    step_size = None
    for f in info["filters"]:
        if f["filterType"] == "LOT_SIZE":
            step_size = float(f["stepSize"])
            break
    price = float(client.get_symbol_ticker(symbol=symbol)["price"])
    qty = usdt_amount / price
    if step_size:
        precision = len(str(step_size).rstrip("0").split(".")[-1])
        qty = round(qty - (qty % step_size), precision)
    return qty


# ─── Signal Generation ────────────────────────────────────
def get_signal(df, model, scaler, confidence_threshold=0.60):
    latest = df[FEATURE_COLS].iloc[-1:]
    latest_scaled = scaler.transform(latest)
    prob = model.predict_proba(latest_scaled)[0]
    buy_prob  = prob[1]
    sell_prob = prob[0]

    log.info(f"📊 BUY confidence: {buy_prob:.2%} | SELL confidence: {sell_prob:.2%}")

    if buy_prob >= confidence_threshold:
        return "BUY", buy_prob
    elif sell_prob >= confidence_threshold:
        return "SELL", sell_prob
    return "HOLD", max(buy_prob, sell_prob)


# ─── Trade Execution ──────────────────────────────────────
class TradeManager:
    def __init__(self, client):
        self.client = client
        self.in_position = False
        self.entry_price = None
        self.entry_qty   = None
        self.daily_trades = 0
        self.daily_pnl    = 0.0
        self.trade_log    = []

    def buy(self, symbol, usdt_budget):
        if self.in_position:
            log.info("⏭️ Already in position, skipping BUY")
            return False

        if self.daily_trades >= CONFIG["max_trades_per_day"]:
            log.warning("🛑 Max daily trades reached")
            return False

        qty = calculate_quantity(self.client, symbol, usdt_budget)
        if qty <= 0:
            log.warning("⚠️ Quantity too small to trade")
            return False

        try:
            order = self.client.order_market_buy(symbol=symbol, quantity=qty)
            self.entry_price = float(order["fills"][0]["price"])
            self.entry_qty   = float(order["executedQty"])
            self.in_position = True
            self.daily_trades += 1
            log.info(f"✅ BUY  | {qty} {symbol} @ ${self.entry_price:.4f}")
            self._log_trade("BUY", symbol, qty, self.entry_price)
            return True
        except BinanceAPIException as e:
            log.error(f"❌ BUY failed: {e}")
            return False

    def sell(self, symbol):
        if not self.in_position:
            log.info("⏭️ Not in position, skipping SELL")
            return False

        base_asset = symbol.replace("USDT", "")
        qty = get_asset_balance(self.client, base_asset)
        qty = round(qty * 0.999, 6)  # Leave tiny buffer for fees

        try:
            order = self.client.order_market_sell(symbol=symbol, quantity=qty)
            exit_price = float(order["fills"][0]["price"])
            pnl = (exit_price - self.entry_price) / self.entry_price * 100
            self.daily_pnl += pnl
            self.in_position  = False
            log.info(f"✅ SELL | {qty} {symbol} @ ${exit_price:.4f} | PnL: {pnl:.2f}%")
            self._log_trade("SELL", symbol, qty, exit_price, pnl)
            self.entry_price = None
            self.entry_qty   = None
            return True
        except BinanceAPIException as e:
            log.error(f"❌ SELL failed: {e}")
            return False

    def check_stop_loss_take_profit(self, current_price, symbol):
        if not self.in_position or self.entry_price is None:
            return

        change = (current_price - self.entry_price) / self.entry_price * 100

        if change <= -CONFIG["stop_loss_pct"]:
            log.warning(f"🔴 Stop-loss triggered! Change: {change:.2f}%")
            self.sell(symbol)

        elif change >= CONFIG["take_profit_pct"]:
            log.info(f"🟢 Take-profit triggered! Change: {change:.2f}%")
            self.sell(symbol)

    def _log_trade(self, side, symbol, qty, price, pnl=None):
        self.trade_log.append({
            "time": datetime.now().isoformat(),
            "side": side, "symbol": symbol,
            "qty": qty, "price": price, "pnl": pnl
        })
        pd.DataFrame(self.trade_log).to_csv("trades.csv", index=False)


# ─── Main Bot Loop ────────────────────────────────────────
def run_bot():
    log.info("=" * 50)
    log.info("🤖 AI Trading Bot Starting...")
    log.info("=" * 50)

    client  = get_client()
    symbol  = CONFIG["symbol"]
    interval = CONFIG["interval"]
    model_path = "model.pkl"

    # Train or load model
    df = fetch_ohlcv(client, symbol, interval)
    df = add_features(df)

    if os.path.exists(model_path):
        log.info("📂 Loading existing model...")
        model, scaler = load_model(model_path)
    else:
        model, scaler = train_model(df.copy(), model_path)

    trader = TradeManager(client)
    retrain_counter = 0

    log.info(f"🎯 Trading {symbol} on {interval} candles")
    log.info(f"💰 Risk per trade: {CONFIG['risk_pct']*100:.0f}% of balance")
    log.info("-" * 50)

    while True:
        try:
            # Fetch latest data
            df = fetch_ohlcv(client, symbol, interval)
            df = add_features(df)

            current_price = df["close"].iloc[-1]
            usdt_balance  = get_usdt_balance(client)

            log.info(f"💵 USDT Balance: ${usdt_balance:.2f} | {symbol}: ${current_price:.4f}")

            # Check stop-loss / take-profit
            trader.check_stop_loss_take_profit(current_price, symbol)

            # Check daily loss limit
            if trader.daily_pnl <= -CONFIG["max_daily_loss_pct"]:
                log.warning(f"🛑 Daily loss limit hit ({trader.daily_pnl:.2f}%). Stopping for today.")
                time.sleep(3600)
                continue

            # Get AI signal
            signal, confidence = get_signal(df, model, scaler, CONFIG["confidence_threshold"])
            log.info(f"🤖 Signal: {signal} (confidence: {confidence:.2%})")

            usdt_to_use = usdt_balance * CONFIG["risk_pct"]

            if signal == "BUY" and usdt_balance >= CONFIG["min_usdt_trade"]:
                trader.buy(symbol, usdt_to_use)

            elif signal == "SELL" and trader.in_position:
                trader.sell(symbol)

            # Retrain every N cycles
            retrain_counter += 1
            if retrain_counter >= CONFIG["retrain_every_n_cycles"]:
                log.info("🔄 Retraining model with fresh data...")
                model, scaler = train_model(df.copy(), model_path)
                retrain_counter = 0

            sleep_time = CONFIG["sleep_seconds"]
            log.info(f"⏳ Sleeping {sleep_time}s until next check...\n")
            time.sleep(sleep_time)

        except BinanceAPIException as e:
            log.error(f"Binance API error: {e}")
            time.sleep(60)

        except KeyboardInterrupt:
            log.info("⛔ Bot stopped by user.")
            if trader.in_position:
                log.warning("⚠️ You still have an open position! Check Binance.")
            break

        except Exception as e:
            log.error(f"Unexpected error: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    run_bot()
