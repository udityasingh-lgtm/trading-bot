"""
============================================================
  AI Trading Bot v2 - Technical + News + Fear & Greed
============================================================
"""

import os
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime
import requests
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

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("bot.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


# ── Binance Client ────────────────────────────────────────
def get_client():
    api_key    = os.getenv("BINANCE_API_KEY",    CONFIG["api_key"])
    api_secret = os.getenv("BINANCE_API_SECRET", CONFIG["api_secret"])
    if CONFIG["testnet"]:
        client = Client(api_key, api_secret, testnet=True)
        client.API_URL = "https://testnet.binance.vision/api"
        log.info("[TESTNET] Connected to Binance Testnet")
    else:
        client = Client(api_key, api_secret)
        log.info("[LIVE] Connected to Binance LIVE")
    return client


# ── Fear & Greed Index ────────────────────────────────────
def get_fear_and_greed():
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        res = requests.get(url, timeout=10)
        data = res.json()
        value = int(data["data"][0]["value"])
        label = data["data"][0]["value_classification"]
        log.info(f"[FEAR&GREED] Score: {value} ({label})")
        return value, label
    except Exception as e:
        log.warning(f"[FEAR&GREED] Failed to fetch: {e}")
        return 50, "Neutral"  # Default to neutral


def fear_greed_signal(score):
    """
    Returns signal based on Fear & Greed score
    Extreme Fear = good time to BUY
    Extreme Greed = good time to SELL
    """
    if score <= 25:
        return "BUY", "Extreme Fear - good time to buy!"
    elif score <= 45:
        return "BUY", "Fear - market may be oversold"
    elif score <= 55:
        return "HOLD", "Neutral - no clear signal"
    elif score <= 75:
        return "SELL", "Greed - market may be overbought"
    else:
        return "SELL", "Extreme Greed - good time to sell!"


# ── Crypto News Sentiment ─────────────────────────────────
def get_news_sentiment(symbol):
    try:
        # Using CryptoCompare free news API
        coin = symbol.replace("USDT", "")
        url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={coin}&limit=10"
        res = requests.get(url, timeout=10)
        articles = res.json().get("Data", [])

        if not articles:
            return "NEUTRAL", 0

        # Simple keyword-based sentiment
        positive_words = [
            "surge", "rally", "bull", "gain", "rise", "pump", "high",
            "record", "growth", "adoption", "partnership", "upgrade",
            "positive", "boost", "breakout", "moon", "buy"
        ]
        negative_words = [
            "crash", "drop", "bear", "loss", "fall", "dump", "low",
            "hack", "ban", "sell", "fear", "decline", "correction",
            "negative", "warning", "risk", "lawsuit", "fraud"
        ]

        pos_count = 0
        neg_count = 0

        for article in articles[:10]:
            title = article.get("title", "").lower()
            body  = article.get("body", "").lower()[:200]
            text  = title + " " + body

            pos_count += sum(1 for w in positive_words if w in text)
            neg_count += sum(1 for w in negative_words if w in text)

        total = pos_count + neg_count
        if total == 0:
            sentiment_score = 0
            sentiment = "NEUTRAL"
        else:
            sentiment_score = (pos_count - neg_count) / total * 100
            if sentiment_score > 20:
                sentiment = "POSITIVE"
            elif sentiment_score < -20:
                sentiment = "NEGATIVE"
            else:
                sentiment = "NEUTRAL"

        log.info(f"[NEWS] Sentiment: {sentiment} (score: {sentiment_score:.1f}, +{pos_count}/-{neg_count})")
        return sentiment, sentiment_score

    except Exception as e:
        log.warning(f"[NEWS] Failed to fetch: {e}")
        return "NEUTRAL", 0


def news_signal(sentiment):
    if sentiment == "POSITIVE":
        return "BUY"
    elif sentiment == "NEGATIVE":
        return "SELL"
    return "HOLD"


# ── Combined Signal ───────────────────────────────────────
def combined_signal(tech_signal, tech_conf, fg_signal, news_signal_val, fg_score, news_sentiment):
    """
    Combines all 3 signals with weights:
    - Technical: 50%
    - Fear & Greed: 30%
    - News: 20%
    """
    # Convert signals to scores
    def signal_to_score(sig):
        if sig == "BUY":  return 1
        if sig == "SELL": return -1
        return 0

    tech_score  = signal_to_score(tech_signal)  * tech_conf * 0.50
    fg_score_w  = signal_to_score(fg_signal)    * 0.30
    news_score  = signal_to_score(news_signal_val) * 0.20

    total_score = tech_score + fg_score_w + news_score

    log.info(f"[COMBINED] Tech: {tech_signal}({tech_conf:.0%}) | F&G: {fg_signal}({fg_score}) | News: {news_signal_val}({news_sentiment})")
    log.info(f"[COMBINED] Total Score: {total_score:.3f}")

    if total_score >= 0.25:
        return "BUY", total_score
    elif total_score <= -0.25:
        return "SELL", abs(total_score)
    return "HOLD", abs(total_score)


# ── Data & Features ───────────────────────────────────────
def fetch_ohlcv(client, symbol, interval, limit=600):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "timestamp","open","high","low","close","volume",
        "close_time","qav","trades","tbbav","tbqav","ignore"
    ])
    for col in ["open","high","low","close","volume"]:
        df[col] = df[col].astype(float)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    return df[["open","high","low","close","volume"]]


def add_features(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    df["ema_9"]       = ta.trend.EMAIndicator(c, 9).ema_indicator()
    df["ema_21"]      = ta.trend.EMAIndicator(c, 21).ema_indicator()
    df["ema_50"]      = ta.trend.EMAIndicator(c, 50).ema_indicator()
    macd              = ta.trend.MACD(c)
    df["macd"]        = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_diff"]   = macd.macd_diff()
    df["adx"]         = ta.trend.ADXIndicator(h, l, c).adx()
    df["rsi"]         = ta.momentum.RSIIndicator(c, 14).rsi()
    df["stoch_k"]     = ta.momentum.StochasticOscillator(h, l, c).stoch()
    df["cci"]         = ta.trend.CCIIndicator(h, l, c).cci()
    df["roc"]         = ta.momentum.ROCIndicator(c, 10).roc()
    bb                = ta.volatility.BollingerBands(c)
    df["bb_upper"]    = bb.bollinger_hband()
    df["bb_lower"]    = bb.bollinger_lband()
    df["bb_width"]    = bb.bollinger_wband()
    df["bb_pct"]      = bb.bollinger_pband()
    df["atr"]         = ta.volatility.AverageTrueRange(h, l, c).average_true_range()
    df["obv"]         = ta.volume.OnBalanceVolumeIndicator(c, v).on_balance_volume()
    df["cmf"]         = ta.volume.ChaikinMoneyFlowIndicator(h, l, c, v).chaikin_money_flow()
    df["returns_1"]   = c.pct_change(1)
    df["returns_3"]   = c.pct_change(3)
    df["hl_ratio"]    = (h - l) / c
    df["oc_ratio"]    = (c - df["open"]) / df["open"]
    df.dropna(inplace=True)
    return df


FEATURE_COLS = [
    "ema_9","ema_21","ema_50","macd","macd_signal","macd_diff","adx",
    "rsi","stoch_k","cci","roc","bb_upper","bb_lower","bb_width",
    "bb_pct","atr","obv","cmf","returns_1","returns_3","hl_ratio","oc_ratio"
]


def train_model(df, save_path="model.pkl"):
    log.info("[AI] Training model...")
    df["target"] = (df["close"].shift(-3) / df["close"] - 1 > 0.005).astype(int)
    df = df.dropna()
    X = df[FEATURE_COLS]
    y = df["target"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=False)
    model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight="balanced")
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    log.info(f"[AI] Model accuracy: {acc:.2%}")
    joblib.dump({"model": model, "scaler": scaler}, save_path)
    return model, scaler


def load_model(path="model.pkl"):
    data = joblib.load(path)
    return data["model"], data["scaler"]


def get_tech_signal(df, model, scaler, threshold=0.60):
    latest = df[FEATURE_COLS].iloc[-1:]
    latest_scaled = scaler.transform(latest)
    prob = model.predict_proba(latest_scaled)[0]
    buy_prob  = prob[1]
    sell_prob = prob[0]
    log.info(f"[TECHNICAL] BUY: {buy_prob:.2%} | SELL: {sell_prob:.2%}")
    if buy_prob >= threshold:
        return "BUY", buy_prob
    elif sell_prob >= threshold:
        return "SELL", sell_prob
    return "HOLD", max(buy_prob, sell_prob)


# ── Trade Manager ─────────────────────────────────────────
def get_usdt_balance(client):
    return float(client.get_asset_balance(asset="USDT")["free"])


def get_asset_balance(client, asset):
    return float(client.get_asset_balance(asset=asset)["free"])


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


class TradeManager:
    def __init__(self, client):
        self.client      = client
        self.positions   = {}
        self.daily_trades = 0
        self.daily_pnl    = 0.0
        self.trade_log    = []

    def buy(self, symbol, usdt_budget):
        if symbol in self.positions:
            log.info(f"[SKIP] Already in {symbol} position")
            return False
        if self.daily_trades >= CONFIG["max_trades_per_day"]:
            log.warning("[LIMIT] Max daily trades reached")
            return False
        qty = calculate_quantity(self.client, symbol, usdt_budget)
        if qty <= 0:
            log.warning("[SKIP] Quantity too small")
            return False
        try:
            order = self.client.order_market_buy(symbol=symbol, quantity=qty)
            entry_price = float(order["fills"][0]["price"])
            self.positions[symbol] = {"price": entry_price, "qty": float(order["executedQty"])}
            self.daily_trades += 1
            log.info(f"[BUY] {qty} {symbol} @ ${entry_price:.4f}")
            self._log_trade("BUY", symbol, qty, entry_price)
            return True
        except BinanceAPIException as e:
            log.error(f"[ERROR] BUY failed: {e}")
            return False

    def sell(self, symbol):
        if symbol not in self.positions:
            log.info(f"[SKIP] Not in {symbol} position")
            return False
        base_asset = symbol.replace("USDT", "")
        qty = round(get_asset_balance(self.client, base_asset) * 0.999, 6)
        try:
            order = self.client.order_market_sell(symbol=symbol, quantity=qty)
            exit_price  = float(order["fills"][0]["price"])
            entry_price = self.positions[symbol]["price"]
            pnl = (exit_price - entry_price) / entry_price * 100
            self.daily_pnl += pnl
            log.info(f"[SELL] {qty} {symbol} @ ${exit_price:.4f} | PnL: {pnl:.2f}%")
            self._log_trade("SELL", symbol, qty, exit_price, pnl)
            del self.positions[symbol]
            return True
        except BinanceAPIException as e:
            log.error(f"[ERROR] SELL failed: {e}")
            return False

    def check_sl_tp(self, symbol, current_price):
        if symbol not in self.positions:
            return
        entry = self.positions[symbol]["price"]
        change = (current_price - entry) / entry * 100
        if change <= -CONFIG["stop_loss_pct"]:
            log.warning(f"[STOPLOSS] {symbol} dropped {change:.2f}%!")
            self.sell(symbol)
        elif change >= CONFIG["take_profit_pct"]:
            log.info(f"[TAKEPROFIT] {symbol} up {change:.2f}%!")
            self.sell(symbol)

    def _log_trade(self, side, symbol, qty, price, pnl=None):
        self.trade_log.append({
            "time": datetime.now().isoformat(),
            "side": side, "symbol": symbol,
            "qty": qty, "price": price, "pnl": pnl
        })
        pd.DataFrame(self.trade_log).to_csv("trades.csv", index=False)


# ── Main Bot Loop ─────────────────────────────────────────
def run_bot():
    log.info("=" * 55)
    log.info("  AI Trading Bot v2 - Technical + News + Fear&Greed")
    log.info("=" * 55)

    client   = get_client()
    symbols  = CONFIG["symbols"]
    interval = CONFIG["interval"]
    model_path = "model.pkl"
    retrain_counter = 0

    # Train model on first symbol
    df = fetch_ohlcv(client, symbols[0], interval)
    df = add_features(df)
    if os.path.exists(model_path):
        log.info("[AI] Loading existing model...")
        model, scaler = load_model(model_path)
    else:
        model, scaler = train_model(df.copy(), model_path)

    trader = TradeManager(client)

    log.info(f"[BOT] Trading: {', '.join(symbols)}")
    log.info(f"[BOT] Interval: {interval}")
    log.info("-" * 55)

    while True:
        try:
            # Get Fear & Greed (same for all coins)
            fg_score, fg_label = get_fear_and_greed()
            fg_sig, fg_reason  = fear_greed_signal(fg_score)

            usdt_balance = get_usdt_balance(client)
            usdt_per_coin = usdt_balance / len(symbols)
            log.info(f"[BALANCE] Total USDT: ${usdt_balance:.2f} | Per coin: ${usdt_per_coin:.2f}")

            # Check daily loss limit
            if trader.daily_pnl <= -CONFIG["max_daily_loss_pct"]:
                log.warning(f"[LIMIT] Daily loss limit hit! Pausing 1 hour...")
                time.sleep(3600)
                continue

            # Loop through each coin
            for symbol in symbols:
                log.info(f"\n--- Analyzing {symbol} ---")

                # Fetch data
                df = fetch_ohlcv(client, symbol, interval)
                df = add_features(df)
                current_price = df["close"].iloc[-1]

                log.info(f"[PRICE] {symbol}: ${current_price:.4f}")

                # Check stop loss / take profit
                trader.check_sl_tp(symbol, current_price)

                # Get news sentiment
                news_sent, news_score = get_news_sentiment(symbol)
                news_sig = news_signal(news_sent)

                # Get technical signal
                tech_sig, tech_conf = get_tech_signal(df, model, scaler, CONFIG["confidence_threshold"])

                # Combine all signals
                final_signal, final_conf = combined_signal(
                    tech_sig, tech_conf, fg_sig, news_sig, fg_score, news_sent
                )

                log.info(f"[FINAL] {symbol}: {final_signal} (strength: {final_conf:.3f})")

                # Execute trade
                if final_signal == "BUY" and usdt_per_coin >= CONFIG["min_usdt_trade"]:
                    trader.buy(symbol, usdt_per_coin * CONFIG["risk_pct"])
                elif final_signal == "SELL":
                    trader.sell(symbol)

            # Retrain model periodically
            retrain_counter += 1
            if retrain_counter >= CONFIG["retrain_every_n_cycles"]:
                log.info("[AI] Retraining model with fresh data...")
                df = fetch_ohlcv(client, symbols[0], interval)
                df = add_features(df)
                model, scaler = train_model(df.copy(), model_path)
                retrain_counter = 0

            log.info(f"\n[WAIT] Sleeping {CONFIG['sleep_seconds']}s...\n")
            time.sleep(CONFIG["sleep_seconds"])

        except BinanceAPIException as e:
            log.error(f"[ERROR] Binance API: {e}")
            time.sleep(60)
        except KeyboardInterrupt:
            log.info("[STOP] Bot stopped by user.")
            break
        except Exception as e:
            log.error(f"[ERROR] Unexpected: {e}", exc_info=True)
            time.sleep(60)


if __name__ == "__main__":
    run_bot()
