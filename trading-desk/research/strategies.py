"""今晚的 5 個假設。每個函式回傳 {sleeve 名稱: 目標持有序列}，在收盤時計算。"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _hold_state(entry: pd.Series, exit_: pd.Series, max_hold: int | None = None) -> pd.Series:
    on, held, out = False, 0, []
    for e, x in zip(entry.fillna(False), exit_.fillna(False)):
        if on:
            held += 1
            if x or (max_hold and held >= max_hold):
                on = False
        elif e:
            on, held = True, 0
        out.append(on)
    return pd.Series(out, index=entry.index)


def _to_target(state: pd.Series, ticker: str, cal: pd.DatetimeIndex) -> pd.Series:
    s = state.reindex(cal).fillna(False).astype(bool)
    return pd.Series(np.where(s, ticker, None), index=cal, dtype=object)


def rsi(close: pd.Series, n: int) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# A. 趨勢＋動量＋量能共振（放量突破）
def trend_volume(frames, cal, vol_mult=1.5):
    out = {}
    for s, df in frames.items():
        c, v = df["close"], df["volume"]
        entry = (c >= c.rolling(20).max()) & (c > c.rolling(60).mean()) & (v > vol_mult * v.rolling(20).mean())
        exit_ = c < c.rolling(20).mean()
        out[s] = _to_target(_hold_state(entry, exit_), s, cal)
    return out


# B. 過度反應後的均值回歸
def mean_revert(frames, cal, rsi_max=10):
    out = {}
    for s, df in frames.items():
        c = df["close"]
        entry = (c > c.rolling(200).mean()) & (rsi(c, 2) < rsi_max)
        exit_ = c > c.rolling(5).mean()
        out[s] = _to_target(_hold_state(entry, exit_, max_hold=10), s, cal)
    return out


# C. 板塊輪動：每月持有相對強度最強的 ETF；0050 跌破 200 日線改持現金
def rotation(frames, cal, lookback=63):
    closes = pd.DataFrame({s: df["close"] for s, df in frames.items()}).reindex(cal).ffill()
    mom = closes / closes.shift(lookback) - 1
    risk_on = closes["0050"] > closes["0050"].rolling(200).mean()
    month_end = pd.Series(cal.to_period("M"), index=cal)
    month_end = month_end != month_end.shift(-1)
    tgt, cur = [], None
    for d in cal:
        if month_end[d]:
            row = mom.loc[d].dropna()
            cur = row.idxmax() if risk_on[d] and len(row) else None
        tgt.append(cur)
    return {"rotation": pd.Series(tgt, index=cal, dtype=object)}


# D. 除息前漂移：除息前 N 個交易日買進，除息前一日開盤賣出（不領息）
def pre_dividend(frames, cal, days=10, ex_dates=None):
    out = {}
    for s, df in frames.items():
        state = pd.Series(False, index=cal)
        for d in ex_dates.get(s, []):
            if d not in cal:
                continue
            k = cal.get_loc(d)
            lo, hi = k - days, k - 2  # 第 k-2 日收盤訊號為空 → 第 k-1 日開盤賣出
            if lo >= 0 and hi > lo:
                state.iloc[lo:hi] = True
        out[s] = _to_target(state, s, cal)
    return out


# E. 波動率切換：0050 的 20 日實現波動率低於自身一年分位數時持有
def vol_switch(frames, cal, q=0.7):
    c = frames["0050"]["close"]
    rv = c.pct_change().rolling(20).std()
    thr = rv.rolling(252, min_periods=126).quantile(q)
    state = rv < thr
    return {"0050": _to_target(state, "0050", cal)}


STRATEGIES = {
    "A_趨勢量能共振": dict(fn=trend_volume, param="vol_mult", base=1.5, grid=[1.2, 1.5, 2.0], stop=2.0, sleeves=True),
    "B_均值回歸": dict(fn=mean_revert, param="rsi_max", base=10, grid=[5, 10, 15], stop=3.0, sleeves=True),
    "C_板塊輪動": dict(fn=rotation, param="lookback", base=63, grid=[21, 63, 126], stop=3.0, sleeves=False),
    "D_除息前漂移": dict(fn=pre_dividend, param="days", base=10, grid=[5, 10, 15], stop=2.0, sleeves=True),
    "E_波動率切換": dict(fn=vol_switch, param="q", base=0.7, grid=[0.5, 0.7, 0.8], stop=3.0, sleeves=False),
}

THESIS = {
    "A_趨勢量能共振": ("創 20 日新高、站上季線且成交量放大 1.5 倍時，趨勢延續的機率較高。",
                  "後知後覺追進的資金與太早獲利了結的人；資訊是逐步擴散的。",
                  "盤整或來回洗盤的市場：假突破多、每次都被 20 日線洗出場。"),
    "B_均值回歸": ("長期多頭中的急跌（RSI(2) 極低）多半是過度反應，數日內會反彈。",
              "恐慌賣出的散戶與被迫停損的人；我們提供流動性、賺取溢酬。",
              "真正的空頭起點：急跌後繼續跌，200 日線過濾來不及反應。"),
    "C_板塊輪動": ("過去一季最強的 ETF 下個月傾向繼續強（動量），大盤跌破 200 日線時躲到現金。",
              "每月才慢速再平衡的資金與追逐落後補漲的人。",
              "動量崩潰：領漲族群突然反轉（如 2022 年科技股），月頻換股來不及。"),
    "D_除息前漂移": ("除息前存股族搶買推升價格；在他們之前買、除息前一天賣給他們。",
                "為了領息而在除息前買進的存股族。",
                "除息行情被提前反映或市場整體下跌時，短期持有吃不到漂移、只付成本。"),
    "E_波動率切換": ("低波動時期趨勢較穩，高波動時期回撤集中；只在低波動時持有 0050。",
                "高波動時被迫去槓桿的資金；低波動時穩定流入的被動資金。",
                "V 型反彈：高波動期常包含最大的反彈日，空手會錯過。"),
}
