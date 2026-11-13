"""台股 ETF 日線資料：FinMind 下載、快取、還原權息（除息＋分割）。

金鑰由雲端環境的網路代理自動帶入 Authorization 標頭，程式內不放任何金鑰。
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://api.finmindtrade.com/api/v4/data"
CACHE = Path(__file__).parent / "data"
UNIVERSE = ["0050", "006208", "0056", "00878", "00713", "0052", "00692", "00850"]
START = "2019-01-01"


def _get(dataset: str, stock_id: str, start: str = START, refresh: bool = False) -> list[dict]:
    """有快取就用快取；refresh=True 時只抓快取最後一天往前 7 天起的新資料並合併（省請求額度）。"""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{dataset}_{stock_id}.json"
    cached = json.loads(path.read_text()) if path.exists() else None
    if cached is not None and not refresh:
        return cached
    since = start
    if cached:
        last = max(r["date"] for r in cached)
        since = (pd.Timestamp(last) - pd.Timedelta(days=7)).date().isoformat()
    q = urllib.parse.urlencode({"dataset": dataset, "data_id": stock_id, "start_date": since})
    with urllib.request.urlopen(f"{API}?{q}", timeout=60) as r:
        payload = json.load(r)
    if payload.get("status") != 200:
        raise RuntimeError(f"FinMind {dataset} {stock_id}: {payload.get('msg')}")
    rows = {(r["date"], r.get("type", "")): r for r in (cached or [])}
    rows.update({(r["date"], r.get("type", "")): r for r in payload["data"]})
    merged = sorted(rows.values(), key=lambda r: r["date"])
    path.write_text(json.dumps(merged))
    time.sleep(0.5)  # 對免費額度客氣一點
    return merged


def load(stock_id: str, refresh: bool = False, refresh_events: bool = False) -> pd.DataFrame:
    """回傳還原後的 open/high/low/close/volume，以及原始收盤 raw_close。
    refresh 只更新價格；除息與分割另由 refresh_events 決定（持有中的標的才需要）。"""
    px = pd.DataFrame(_get("TaiwanStockPrice", stock_id, refresh=refresh))
    if px.empty:
        raise RuntimeError(f"{stock_id}: 沒有價格資料")
    px["date"] = pd.to_datetime(px["date"])
    px = px.set_index("date").sort_index()
    px = px[(px["close"] > 0) & (px["Trading_Volume"] > 0)]  # 去掉停牌／無成交日
    df = pd.DataFrame({
        "open": px["open"], "high": px["max"], "low": px["min"],
        "close": px["close"], "volume": px["Trading_Volume"].astype(float),
    })
    df["raw_close"] = df["close"]

    events = []  # (生效日, 價格比例, 是否為分割)
    for r in _get("TaiwanStockDividendResult", stock_id, refresh=refresh_events):
        if r["before_price"] > 0 and r["after_price"] > 0:
            events.append((pd.Timestamp(r["date"]), r["after_price"] / r["before_price"], False))
    for r in _get("TaiwanStockSplitPrice", stock_id, refresh=refresh_events):
        if r["before_price"] > 0 and r["after_price"] > 0:
            events.append((pd.Timestamp(r["date"]), r["after_price"] / r["before_price"], True))

    factor = pd.Series(1.0, index=df.index)
    split_factor = pd.Series(1.0, index=df.index)
    for d, ratio, is_split in events:
        before = df.index < d
        factor[before] *= ratio
        if is_split:
            split_factor[before] *= ratio
    for c in ("open", "high", "low", "close"):
        df[c] = df[c] * factor
    df["volume"] = df["volume"] / split_factor  # 分割前成交量換算成分割後股數
    df.attrs["events"] = [(d.date().isoformat(), round(x, 5), s) for d, x, s in events]
    return df


def load_universe(refresh: bool = False) -> dict[str, pd.DataFrame]:
    return {s: load(s, refresh) for s in UNIVERSE}


if __name__ == "__main__":
    for s, df in load_universe().items():
        print(s, df.index[0].date(), "→", df.index[-1].date(), len(df), "rows,",
              len(df.attrs["events"]), "events, last raw close", df["raw_close"].iloc[-1])
