"""模擬交易機器人：策略 A（趨勢＋量能共振），台股 ETF，零股，自建帳本。

每個交易日收盤後執行一次（資料約 15:00 後更新）：
  python3 trading-desk/paper/paper.py run      # 撮合昨天掛的單 → 停損 → 熔斷 → 今天收盤產生明天的單
  python3 trading-desk/paper/paper.py status   # 印出帳戶狀態（晨報用）

流程：圖表分析師（訊號評分）→ 風控官（否決）→ 交易員（掛限價單，唯一會「下單」的函式）。
所有價格均為真實收盤資料；成交規則：
  買進限價單：開盤 ≤ 限價 → 以開盤價成交；否則最低價 ≤ 限價 → 以限價成交；否則當日失效。
  賣出（訊號出場）：次日開盤市價。停損：最低價觸及即出場，跳空則以開盤價。
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "research"))
import data  # noqa: E402

STATE = HERE / "state.json"
LEDGER = HERE / "ledger.csv"
JOURNAL = HERE.parent / "journal"

CAPITAL = 200_000
RISK = 0.02          # 單筆風險（佔該 sleeve 權益）
STOP_MULT = 2.0      # 停損 = 進場價 − 2 × ATR14
VOL_MULT = 1.5
LIMIT_BUFFER = 0.005  # 限價 = 收盤 × (1 + 0.5%)
DAILY_HALT = 0.02    # 單日虧損 2% 熔斷
MAX_LOSS_STREAK = 3
FEE, TAX, SLIP = 0.001425, 0.001, 0.0005


def tick(px: float) -> float:
    """ETF 升降單位：50 元以下 0.01，以上 0.05。"""
    t = 0.01 if px < 50 else 0.05
    return round(round(px / t) * t, 2)


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return dict(cash=CAPITAL, positions={}, orders=[], last_run=None, equity_hist=[],
                loss_streak=0, halted_until=None, halt_reason=None)


def raw_frame(df: pd.DataFrame) -> pd.DataFrame:
    f = df["raw_close"] / df["close"]
    out = df[["open", "high", "low", "close"]].mul(f, axis=0)
    out["volume"] = df["volume"]
    return out


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


# ---------- 圖表分析師 ----------
def analyst(df: pd.DataFrame) -> dict:
    """在最後一根 K 線上評估策略 A 的形態，1–10 分；≥7 分才放行。"""
    c, v = df["close"], df["volume"]
    checks = {
        "創20日新高": c.iloc[-1] >= c.rolling(20).max().iloc[-1],
        "站上60日線": c.iloc[-1] > c.rolling(60).mean().iloc[-1],
        "量增1.5倍": v.iloc[-1] > VOL_MULT * v.rolling(20).mean().iloc[-1],
        "20日報酬為正": c.iloc[-1] > c.iloc[-21],
        "60日線上彎": c.rolling(60).mean().iloc[-1] > c.rolling(60).mean().iloc[-6],
    }
    core = checks["創20日新高"] and checks["站上60日線"] and checks["量增1.5倍"]
    score = (6 if core else 0) + 2 * checks["20日報酬為正"] + 2 * checks["60日線上彎"]
    if not core:
        score = sum(checks.values()) * 1  # 未達核心條件，最多 5 分
    exit_sig = c.iloc[-1] < c.rolling(20).mean().iloc[-1]
    return dict(score=int(score), checks={k: bool(x) for k, x in checks.items()}, entry=bool(core and score >= 7),
                exit=bool(exit_sig))


# ---------- 風控官 ----------
def risk_officer(st: dict, today: pd.Timestamp, latest: pd.Timestamp, cal: pd.DatetimeIndex) -> list[str]:
    vetoes = []
    if st.get("halted_until") and pd.Timestamp(st["halted_until"]) > today:
        vetoes.append(f"熔斷中至 {st['halted_until']}：{st.get('halt_reason')}")
    if st["loss_streak"] >= MAX_LOSS_STREAK:
        vetoes.append(f"連虧 {st['loss_streak']} 筆，暫停新單，待使用者複查")
    if (pd.Timestamp(date.today()) - latest).days > 4:
        vetoes.append(f"資料過期：最新收盤 {latest.date()}")
    return vetoes


def log_trade(row: dict):
    new = not LEDGER.exists()
    with LEDGER.open("a") as f:
        if new:
            f.write("date,ticker,side,shares,price,cash_flow,reason\n")
        f.write(",".join(str(row[k]) for k in ["date", "ticker", "side", "shares", "price", "cash_flow", "reason"]) + "\n")


def run():
    st = load_state()
    frames = {s: data.load(s, refresh=True) for s in data.UNIVERSE}
    raw = {s: raw_frame(df) for s, df in frames.items()}
    cal = frames["0050"].index
    today = cal[-1]
    if st["last_run"] and pd.Timestamp(st["last_run"]) >= today:
        print(f"{today.date()} 已處理過，略過。"); return
    log = []
    prev_equity = st["equity_hist"][-1][1] if st["equity_hist"] else CAPITAL
    realized = []

    # 1) 交易員：撮合昨天掛的單（只處理今天這根 K 線）
    for o in st["orders"]:
        bar = raw[o["ticker"]].loc[today] if today in raw[o["ticker"]].index else None
        if bar is None:
            log.append(f"{o['ticker']} 今日無成交資料，訂單失效"); continue
        if o["side"] == "buy":
            fill = bar["open"] if bar["open"] <= o["limit"] else (o["limit"] if bar["low"] <= o["limit"] else None)
            if fill is None:
                log.append(f"{o['ticker']} 限價 {o['limit']} 未成交（最低 {bar['low']:.2f}）"); continue
            cost = o["shares"] * fill * (1 + SLIP) * (1 + FEE)
            if cost > st["cash"]:
                log.append(f"{o['ticker']} 現金不足，取消"); continue
            st["cash"] -= cost
            st["positions"][o["ticker"]] = dict(shares=o["shares"], entry=fill, cost=cost, stop=o["stop"],
                                                date=str(today.date()))
            log_trade(dict(date=today.date(), ticker=o["ticker"], side="buy", shares=o["shares"], price=fill,
                           cash_flow=-round(cost), reason="限價進場"))
            log.append(f"買進 {o['ticker']} {o['shares']} 股 @ {fill:.2f}，停損 {o['stop']}")
        else:
            p = st["positions"].pop(o["ticker"], None)
            if p:
                proceeds = p["shares"] * bar["open"] * (1 - SLIP) * (1 - FEE - TAX)
                st["cash"] += proceeds
                realized.append(proceeds - p["cost"])
                log_trade(dict(date=today.date(), ticker=o["ticker"], side="sell", shares=p["shares"],
                               price=bar["open"], cash_flow=round(proceeds), reason="跌破20日線出場"))
                log.append(f"賣出 {o['ticker']} @ {bar['open']:.2f}，損益 {proceeds - p['cost']:+,.0f}")
    st["orders"] = []

    # 1b) 持有期間的除息與分割：股利入帳、股數與停損價調整
    for t, p in st["positions"].items():
        for r in data._get("TaiwanStockDividendResult", t):
            if pd.Timestamp(r["date"]) == today and r["stock_or_cache_dividend"] == "息":
                cash_div = p["shares"] * r["stock_and_cache_dividend"]
                st["cash"] += cash_div
                log_trade(dict(date=today.date(), ticker=t, side="dividend", shares=p["shares"],
                               price=r["stock_and_cache_dividend"], cash_flow=round(cash_div), reason="現金股利"))
                log.append(f"{t} 除息，股利 {cash_div:,.0f} 元入帳（未扣稅與二代健保）")
        for r in data._get("TaiwanStockSplitPrice", t):
            if pd.Timestamp(r["date"]) == today:
                k = r["before_price"] / r["after_price"]
                p["shares"] = int(round(p["shares"] * k))
                p["stop"] = tick(p["stop"] / k)
                p["entry"] = p["entry"] / k
                log.append(f"{t} 分割 1 拆 {k:.0f}，股數改為 {p['shares']}，停損改為 {p['stop']}")

    # 2) 停損
    for t, p in list(st["positions"].items()):
        if today in raw[t].index and raw[t].at[today, "low"] <= p["stop"]:
            px = min(raw[t].at[today, "open"], p["stop"])
            proceeds = p["shares"] * px * (1 - SLIP) * (1 - FEE - TAX)
            st["cash"] += proceeds
            realized.append(proceeds - p["cost"])
            del st["positions"][t]
            log_trade(dict(date=today.date(), ticker=t, side="sell", shares=p["shares"], price=px,
                           cash_flow=round(proceeds), reason="停損"))
            log.append(f"停損 {t} @ {px:.2f}，損益 {proceeds - p['cost']:+,.0f}")

    for pnl in realized:
        st["loss_streak"] = st["loss_streak"] + 1 if pnl < 0 else 0

    equity = st["cash"] + sum(p["shares"] * raw[t]["close"].iloc[-1] for t, p in st["positions"].items())
    day_ret = equity / prev_equity - 1
    if day_ret <= -DAILY_HALT:
        nxt = cal[-1] + pd.offsets.BDay(1)  # 只停下一個交易日的新單
        st["halted_until"], st["halt_reason"] = str(nxt.date()), f"單日虧損 {day_ret:.1%}"
    st["equity_hist"].append([str(today.date()), round(equity, 0)])

    # 3) 分析師 → 風控官 → 交易員：產生明天的單
    vetoes = risk_officer(st, today, today, cal)
    ideas = []
    for s, df in frames.items():
        a = analyst(df)
        r = raw[s]
        if s in st["positions"]:
            if a["exit"]:
                st["orders"].append(dict(ticker=s, side="sell"))
                ideas.append(f"{s}：收盤跌破 20 日線 → 明日開盤賣出")
            continue
        if not a["entry"]:
            continue
        a14 = atr(r).iloc[-1]
        sleeve = equity / len(frames)
        px = r["close"].iloc[-1]
        limit = tick(px * (1 + LIMIT_BUFFER))
        shares = int(min(RISK * sleeve / (STOP_MULT * a14), sleeve / (limit * (1 + SLIP) * (1 + FEE)), st["cash"] / (limit * 1.002)))
        idea = f"{s}：形態 {a['score']}/10（{', '.join(k for k, x in a['checks'].items() if x)}）"
        if vetoes:
            ideas.append(idea + f" → **風控否決**：{'；'.join(vetoes)}"); continue
        if shares <= 0:
            ideas.append(idea + " → 現金不足，略過"); continue
        stop = tick(limit - STOP_MULT * a14)
        st["orders"].append(dict(ticker=s, side="buy", shares=shares, limit=limit, stop=stop))
        ideas.append(idea + f" → 明日限價買進 {shares} 股 @ {limit}，停損 {stop}（風險約 {shares * (limit - stop):,.0f} 元）")

    st["last_run"] = str(today.date())
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))

    # 4) 交易日誌
    JOURNAL.mkdir(exist_ok=True)
    j = JOURNAL / f"{today.date()}.md"
    lines = [f"\n## 模擬盤（策略 A）收盤處理 {today.date()}", "",
             f"- **標的**：{'、'.join(st['positions']) or '空手'}",
             f"- **想法**：{'；'.join(ideas) or '今日無新訊號'}",
             f"- **證據**：{'；'.join(log) or '無成交'}；權益 {equity:,.0f}（當日 {day_ret:+.2%}）",
             f"- **下一步**：{len(st['orders'])} 張待執行訂單" + (f"；風控：{'；'.join(vetoes)}" if vetoes else "")]
    with j.open("a") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def status():
    st = load_state()
    eq = st["equity_hist"][-1] if st["equity_hist"] else [None, CAPITAL]
    print(json.dumps(dict(equity=eq, cash=round(st["cash"]), positions=st["positions"], orders=st["orders"],
                          loss_streak=st["loss_streak"], halted_until=st["halted_until"],
                          total_return=f"{eq[1] / CAPITAL - 1:+.2%}"), ensure_ascii=False, indent=1))


if __name__ == "__main__":
    {"run": run, "status": status}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
