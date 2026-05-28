#!/usr/bin/env python3
"""ETH 以太坊綜合分析腳本 — 技術指標、價格預測、鏈上數據、情緒分析"""

import os
import sys
import requests
import pandas as pd
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from datetime import datetime
from dotenv import load_dotenv

import pandas_ta as ta
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

load_dotenv()

# ── 設定 ─────────────────────────────────────────────────────────────────────
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
DAYS = 365  # 歷史資料天數（需 ≥ 99 才能計算 MA99）

# ── 資料抓取 ──────────────────────────────────────────────────────────────────

def fetch_ohlcv(days: int = DAYS) -> pd.DataFrame:
    """CoinGecko market_chart 日線資料（免 API key，365天可算 MA99）"""
    print("  → 抓取 OHLCV…")
    url = "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
    try:
        r = requests.get(url, params={"vs_currency": "usd", "days": days, "interval": "daily"}, timeout=15)
        r.raise_for_status()
        data = r.json()

        prices  = pd.DataFrame(data["prices"],        columns=["ts", "close"])
        volumes = pd.DataFrame(data["total_volumes"],  columns=["ts", "volume"])

        df = prices.copy()
        df["volume"] = volumes["volume"].values
        df.index = pd.to_datetime(df.pop("ts"), unit="ms").dt.normalize()
        df = df[~df.index.duplicated(keep="last")]

        # 用前後收盤估算 open/high/low（圖表用，指標只用 close）
        df["open"]  = df["close"].shift(1).fillna(df["close"])
        df["high"]  = df[["close", "open"]].max(axis=1) * 1.005
        df["low"]   = df[["close", "open"]].min(axis=1) * 0.995

        return df.dropna()
    except Exception as e:
        print(f"[錯誤] 無法取得價格資料: {e}")
        sys.exit(1)


def fetch_onchain() -> dict:
    """Etherscan — Gas 費用 & ETH 供應量（需 ETHERSCAN_API_KEY）"""
    if not ETHERSCAN_API_KEY:
        return {"_skip": True, "提示": "未設定 ETHERSCAN_API_KEY，跳過鏈上數據"}

    print("  → 抓取鏈上數據…")
    base = "https://api.etherscan.io/v2/api"
    common = {"chainid": "1", "apikey": ETHERSCAN_API_KEY}
    out = {}
    try:
        r = requests.get(base, params={**common, "module": "gastracker", "action": "gasoracle"}, timeout=10)
        g = r.json().get("result", {})
        if isinstance(g, dict):
            out["Gas 慢速 (Gwei)"] = g.get("SafeGasPrice", "N/A")
            out["Gas 標準 (Gwei)"] = g.get("ProposeGasPrice", "N/A")
            out["Gas 快速 (Gwei)"] = g.get("FastGasPrice", "N/A")
        else:
            out["Gas"] = str(g)

        r2 = requests.get(base, params={**common, "module": "stats", "action": "ethsupply"}, timeout=10)
        result2 = r2.json().get("result", "0")
        supply = int(result2) if isinstance(result2, str) else int(result2.get("EthSupply", 0))
        out["ETH 總供應量"] = f"{supply / 1e18 / 1e6:.2f} M ETH"
    except Exception as e:
        out["錯誤"] = str(e)
    return out


def fetch_defi_tvl() -> dict:
    """DeFiLlama — Ethereum TVL（免 API key）"""
    print("  → 抓取 DeFi TVL…")
    try:
        r = requests.get("https://api.llama.fi/v2/chains", timeout=10)
        eth = next((c for c in r.json() if c.get("name", "").lower() == "ethereum"), None)
        if eth:
            return {"Ethereum TVL": f"${eth.get('tvl', 0) / 1e9:.2f} B"}
    except Exception:
        pass
    return {"錯誤": "無法取得 DeFi 數據"}


def fetch_sentiment() -> dict:
    """情緒指標 — Fear & Greed Index"""
    print("  → 抓取市場情緒…")
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=10)
        d = r.json()["data"][0]
        return {"來源": "Fear & Greed Index",
                "數值": d["value"],
                "狀態": d["value_classification"]}
    except Exception:
        return {"錯誤": "無法取得情緒數據"}

# ── 技術分析 ──────────────────────────────────────────────────────────────────

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df.ta.sma(length=7, append=True)
    df.ta.sma(length=25, append=True)
    df.ta.sma(length=99, append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    return df


def read_signals(df: pd.DataFrame) -> list[tuple]:
    """回傳 [(指標名, 訊號說明, 顏色標記, 數值)] 列表"""
    row = df.iloc[-1]
    price = row["close"]
    results = []

    rsi = row.get("RSI_14", 50)
    if rsi > 70:
        results.append(("RSI", "超買  警戒", "🔴", f"{rsi:.1f}"))
    elif rsi < 30:
        results.append(("RSI", "超賣  買點?", "🟢", f"{rsi:.1f}"))
    else:
        results.append(("RSI", "中性", "🟡", f"{rsi:.1f}"))

    macd  = row.get("MACD_12_26_9", 0)
    msig  = row.get("MACDs_12_26_9", 0)
    results.append(("MACD", "多頭交叉" if macd > msig else "空頭交叉",
                    "🟢" if macd > msig else "🔴", f"{macd:.2f}"))

    ma7, ma25, ma99 = row.get("SMA_7", price), row.get("SMA_25", price), row.get("SMA_99", price)
    above = sum([price > ma7, price > ma25, price > ma99])
    results.append(("均線", f"站上 {above}/3 條均線",
                    "🟢" if above >= 2 else "🔴", f"{above}/3"))

    bbu = row.get("BBU_20_2.0", price * 1.05)
    bbl = row.get("BBL_20_2.0", price * 0.95)
    pct = (price - bbl) / max(bbu - bbl, 1) * 100
    if price > bbu:
        label = "突破上軌"
    elif price < bbl:
        label = "跌破下軌"
    else:
        label = f"帶內 {pct:.0f}%"
    results.append(("布林帶", label, "🟡", f"{pct:.0f}%"))

    return results

# ── 價格預測 ──────────────────────────────────────────────────────────────────

def predict(df: pd.DataFrame, horizon: int = 7) -> dict:
    """線性回歸預測（簡單趨勢外推，MAPE 供參考）"""
    prices = df["close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices)

    X = np.arange(len(scaled)).reshape(-1, 1)
    split = int(len(X) * 0.8)
    model = LinearRegression().fit(X[:split], scaled[:split].ravel())

    test_pred = scaler.inverse_transform(model.predict(X[split:]).reshape(-1, 1)).ravel()
    test_real = prices[split:].ravel()
    mape = mean_absolute_percentage_error(test_real, test_pred) * 100

    future_X  = np.arange(len(scaled), len(scaled) + horizon).reshape(-1, 1)
    future_px = scaler.inverse_transform(model.predict(future_X).reshape(-1, 1)).ravel()

    current   = float(prices[-1].item())
    predicted = float(future_px[-1].item())
    chg       = (predicted - current) / current * 100

    return {"current": current, "predicted": predicted, "change_pct": chg,
            "mape": mape, "forecast": future_px,
            "direction": "📈 看漲" if chg > 0 else "📉 看跌"}

# ── 視覺化 ────────────────────────────────────────────────────────────────────

def plot(df: pd.DataFrame, pred: dict, out: str = "eth_analysis.png"):
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(16, 12), facecolor="#0d1117")
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45, height_ratios=[3, 1, 1, 1])
    axes = [fig.add_subplot(gs[i]) for i in range(4)]
    for ax in axes:
        ax.set_facecolor("#161b22")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.tick_params(colors="#8b949e", labelsize=8)

    dates = df.index
    ax1, ax2, ax3, ax4 = axes

    # ── 價格 + MA + BB ──
    ax1.plot(dates, df["close"], color="#58a6ff", lw=1.5, label="ETH Close", zorder=5)
    for col, color, label in [("SMA_7", "#ff7b72", "MA7"), ("SMA_25", "#ffa657", "MA25"), ("SMA_99", "#d2a8ff", "MA99")]:
        if col in df.columns:
            ax1.plot(dates, df[col], color=color, lw=1, alpha=0.8, label=label)
    if "BBU_20_2.0" in df.columns:
        ax1.fill_between(dates, df["BBU_20_2.0"], df["BBL_20_2.0"], alpha=0.08, color="#58a6ff")
        ax1.plot(dates, df["BBU_20_2.0"], color="#58a6ff", lw=0.5, alpha=0.4, ls="--")
        ax1.plot(dates, df["BBL_20_2.0"], color="#58a6ff", lw=0.5, alpha=0.4, ls="--")

    # 預測段
    last_date = dates[-1]
    fdates = pd.date_range(start=last_date, periods=len(pred["forecast"]) + 1, freq="D")[1:]
    ax1.plot(fdates, pred["forecast"], color="#3fb950", lw=1.5, ls="--", alpha=0.9, label="Forecast 7d")
    ax1.axvline(last_date, color="#8b949e", lw=0.5, ls=":")
    ax1.set_title(f"ETH/USD  |  Current: ${pred['current']:,.2f}  |  7d Forecast: ${pred['predicted']:,.2f} ({pred['change_pct']:+.1f}%)",
                  color="white", fontsize=13, fontweight="bold", pad=10)
    ax1.legend(loc="upper left", fontsize=8, framealpha=0.25)
    ax1.set_ylabel("USD", color="#8b949e")

    # ── 成交量 ──
    colors = ["#3fb950" if c >= o else "#f85149" for c, o in zip(df["close"], df["open"])]
    ax2.bar(dates, df["volume"], color=colors, alpha=0.7, width=0.8)
    ax2.set_ylabel("Volume", color="#8b949e", fontsize=9)

    # ── RSI ──
    if "RSI_14" in df.columns:
        ax3.plot(dates, df["RSI_14"], color="#ffa657", lw=1.2)
        ax3.axhline(70, color="#f85149", lw=0.7, ls="--", alpha=0.6)
        ax3.axhline(30, color="#3fb950", lw=0.7, ls="--", alpha=0.6)
        ax3.fill_between(dates, df["RSI_14"], 70, where=df["RSI_14"] > 70, color="#f85149", alpha=0.15)
        ax3.fill_between(dates, df["RSI_14"], 30, where=df["RSI_14"] < 30, color="#3fb950", alpha=0.15)
        ax3.set_ylim(0, 100)
        ax3.set_ylabel("RSI", color="#8b949e", fontsize=9)
        ax3.annotate(f"{df['RSI_14'].iloc[-1]:.1f}", xy=(dates[-1], df["RSI_14"].iloc[-1]),
                     xytext=(-45, 8), textcoords="offset points", color="#ffa657", fontsize=8)

    # ── MACD ──
    if "MACD_12_26_9" in df.columns:
        macd, sig = df["MACD_12_26_9"], df["MACDs_12_26_9"]
        hist = macd - sig
        bar_colors = ["#3fb950" if v >= 0 else "#f85149" for v in hist]
        ax4.bar(dates, hist, color=bar_colors, alpha=0.7, width=0.8)
        ax4.plot(dates, macd, color="#58a6ff", lw=1, label="MACD")
        ax4.plot(dates, sig,  color="#ff7b72", lw=1, label="Signal")
        ax4.axhline(0, color="#8b949e", lw=0.5)
        ax4.set_ylabel("MACD", color="#8b949e", fontsize=9)
        ax4.legend(loc="upper left", fontsize=7, framealpha=0.25)

    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  → 圖表儲存: {out}")

# ── 市場建議 & 分析心得 ───────────────────────────────────────────────────────

def generate_advice(df: pd.DataFrame, signals: list, pred: dict, sentiment: dict) -> dict:
    """根據各項指標自動產生建議與心得"""
    sig_map   = {s[0]: s for s in signals}  # {指標名: (name, label, icon, val)}
    rsi_val   = float(sig_map["RSI"][3])
    macd_bull = sig_map["MACD"][2] == "🟢"
    ma_above  = int(sig_map["均線"][3].split("/")[0])
    fg_val    = int(sentiment.get("數值", 50))
    chg_pct   = pred["change_pct"]

    # ── 綜合評分（-100 ~ +100）──
    score = 0
    score += 30 if rsi_val < 30 else (-30 if rsi_val > 70 else 0)
    score += 20 if macd_bull else -20
    score += (ma_above - 1) * 15           # -15 / 0 / +15 / +30
    score += 20 if fg_val < 25 else (-20 if fg_val > 75 else 0)
    score += 15 if chg_pct > 10 else (-15 if chg_pct < -10 else 0)

    # ── 方向判斷 ──
    if score >= 40:
        stance, icon = "積極做多", "🟢"
    elif score >= 15:
        stance, icon = "謹慎偏多", "🟡"
    elif score >= -15:
        stance, icon = "觀望為主", "⚪"
    elif score >= -40:
        stance, icon = "謹慎偏空", "🟠"
    else:
        stance, icon = "積極做空 / 離場", "🔴"

    # ── 個別建議 ──
    tips = []
    if rsi_val < 30:
        tips.append("RSI 進入超賣區（<30），短線反彈機率上升，可留意止損點位")
    elif rsi_val > 70:
        tips.append("RSI 超買（>70），獲利了結或縮減倉位是合理選擇")
    else:
        tips.append(f"RSI {rsi_val:.1f} 處於中性區間，無明顯超買超賣訊號")

    if not macd_bull:
        tips.append("MACD 空頭交叉，短期動能偏弱，追多需謹慎")
    else:
        tips.append("MACD 多頭交叉，短期上漲動能強，可考慮持多")

    if ma_above == 0:
        tips.append("價格跌破全部均線（MA7/25/99），趨勢偏空，避免重倉做多")
    elif ma_above == 3:
        tips.append("價格站穩全部均線上方，趨勢健康，可順勢持有")
    else:
        tips.append(f"價格站上 {ma_above}/3 條均線，趨勢轉換中，等待方向確認")

    if fg_val < 25:
        tips.append(f"恐懼貪婪 {fg_val}（極度恐懼），歷史上往往是中長線買點，但需等趨勢確認")
    elif fg_val > 75:
        tips.append(f"恐懼貪婪 {fg_val}（極度貪婪），市場過熱，注意回調風險")

    # ── 分析心得 ──
    price_now = df["close"].iloc[-1]
    ma7  = df.get("SMA_7",  pd.Series([price_now])).iloc[-1]
    ma99 = df.get("SMA_99", pd.Series([price_now])).iloc[-1]
    spread = (price_now - ma99) / ma99 * 100

    insight_lines = []
    insight_lines.append(
        f"目前 ETH 報 ${price_now:,.0f}，距 MA99 偏離 {spread:+.1f}%。"
    )
    if spread < -15:
        insight_lines.append(
            "價格大幅低於長期均線，空方力道強勁，但也代表估值相對便宜，"
            "適合分批布局而非單筆重倉。"
        )
    elif spread > 15:
        insight_lines.append(
            "價格大幅高於長期均線，市場情緒偏樂觀，追高風險較高，"
            "可等回測均線再考慮加倉。"
        )
    else:
        insight_lines.append("價格圍繞長期均線運行，多空拉鋸，方向尚不明朗。")

    if fg_val < 30 and rsi_val < 35:
        insight_lines.append(
            "「雙重恐懼」訊號（情緒恐懼 + RSI 超賣）同時出現，"
            "歷史上是風險回報比較佳的進場時機，但需設好停損。"
        )

    insight_lines.append(
        f"線性回歸預測 7 日後為 ${pred['predicted']:,.0f}（{pred['change_pct']:+.1f}%），"
        f"模型誤差 MAPE {pred['mape']:.1f}%，僅反映近期線性趨勢，"
        "實際走勢受消息面影響甚大，請勿單靠此數字操作。"
    )

    return {
        "score":   score,
        "stance":  stance,
        "icon":    icon,
        "tips":    tips,
        "insight": " ".join(insight_lines),
    }

# ── 報告輸出 ──────────────────────────────────────────────────────────────────

def print_report(df, signals, pred, onchain, defi, sentiment) -> str:
    price  = df["close"].iloc[-1]
    prev   = df["close"].iloc[-2]
    chg1d  = (price - prev) / prev * 100

    lines = []
    def p(s=""):
        print(s)
        lines.append(s)

    sep = "─" * 50
    p(f"\n{'═'*50}")
    p(f"  ETH 以太坊綜合分析報告   {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    p(f"{'═'*50}")
    p(f"\n  當前價格  ${price:>10,.2f} USD   ({chg1d:+.2f}% 24h)")

    p(f"\n{sep}")
    p("  技術指標訊號")
    p(sep)
    for name, label, icon, val in signals:
        p(f"  {icon}  {name:<8} {label:<18} {val}")

    p(f"\n{sep}")
    p("  價格預測（線性回歸，僅供參考）")
    p(sep)
    p(f"  當前價格   ${pred['current']:>10,.2f}")
    p(f"  7 日預測   ${pred['predicted']:>10,.2f}  ({pred['change_pct']:+.2f}%)")
    p(f"  方向       {pred['direction']}")
    p(f"  模型誤差   MAPE {pred['mape']:.1f}%")

    p(f"\n{sep}")
    p("  鏈上 / DeFi 數據")
    p(sep)
    if onchain.get("_skip"):
        p(f"  ⚠  {onchain['提示']}")
    else:
        for k, v in onchain.items():
            if not k.startswith("_"):
                p(f"  {k:<22} {v}")
    for k, v in defi.items():
        p(f"  {k:<22} {v}")

    p(f"\n{sep}")
    p("  市場情緒")
    p(sep)
    if "錯誤" in sentiment:
        p(f"  ⚠  {sentiment['錯誤']}")
    elif "數值" in sentiment:
        val = int(sentiment["數值"])
        bar = "█" * (val // 10) + "░" * (10 - val // 10)
        p(f"  Fear & Greed  [{bar}] {val}/100")
        p(f"  狀態          {sentiment['狀態']}")
    else:
        for k, v in sentiment.items():
            p(f"  {k:<22} {v}")

    advice = generate_advice(df, signals, pred, sentiment)

    p(f"\n{sep}")
    p("  綜合市場建議")
    p(sep)
    score_bar_len = min(abs(advice["score"]) // 5, 10)
    score_bar = ("+" if advice["score"] >= 0 else "-") * score_bar_len
    p(f"  {advice['icon']}  立場：{advice['stance']}   評分：{advice['score']:+d}/100  [{score_bar:<10}]")
    p()
    for i, tip in enumerate(advice["tips"], 1):
        words = tip
        while len(words) > 46:
            p(f"  {i}. {words[:46]}")
            words = "     " + words[46:]
            i = " "
        p(f"  {i}. {words}")

    p(f"\n{sep}")
    p("  分析心得")
    p(sep)
    insight = advice["insight"]
    while len(insight) > 46:
        p(f"  {insight[:46]}")
        insight = insight[46:]
    p(f"  {insight}")

    p(f"\n{'═'*50}")
    p("  ⚠  本分析僅供學習參考，不構成投資建議")
    p(f"{'═'*50}\n")

    return "\n".join(lines)

# ── Email 通知 ────────────────────────────────────────────────────────────────

def send_email(report_text: str, chart_path: str = "eth_analysis.png"):
    """透過 Gmail SMTP 寄出分析報告（附圖表）"""
    gmail_addr = os.getenv("GMAIL_ADDRESS", "")
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD", "").replace(" ", "")
    if not gmail_addr or not gmail_pass:
        print("  → 未設定 Gmail 憑證，跳過寄信")
        return

    print(f"  → 寄送 Email 至 {gmail_addr}…")
    msg = MIMEMultipart()
    msg["From"]    = gmail_addr
    msg["To"]      = gmail_addr
    msg["Subject"] = f"ETH 每日分析報告 {datetime.now().strftime('%Y-%m-%d')}"
    msg.attach(MIMEText(report_text, "plain", "utf-8"))

    try:
        with open(chart_path, "rb") as f:
            img = MIMEImage(f.read())
            img.add_header("Content-Disposition", "attachment", filename="eth_analysis.png")
            msg.attach(img)
    except FileNotFoundError:
        pass

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_addr, gmail_pass)
        server.send_message(msg)
    print("  → Email 已寄出 ✓")

# ── 主程式 ────────────────────────────────────────────────────────────────────

def main():
    print("\n[ETH 分析腳本啟動]")
    df        = fetch_ohlcv()
    df        = add_indicators(df)
    signals   = read_signals(df)
    pred      = predict(df)
    onchain   = fetch_onchain()
    defi      = fetch_defi_tvl()
    sentiment = fetch_sentiment()
    print("  → 生成圖表…")
    plot(df, pred)
    report = print_report(df, signals, pred, onchain, defi, sentiment)
    send_email(report)

if __name__ == "__main__":
    main()
