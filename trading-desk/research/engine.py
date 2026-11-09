"""回測引擎：收盤產生訊號、下一根開盤成交、ATR 停損、台股成本。

規則（與 DESK_RULES.md 一致）：
- 手續費 0.1425%（買賣各一次），ETF 證交稅 0.1%（賣出），滑點每邊 0.05%。
- 單筆風險 = 帳戶 2%：股數 = 風險金額 ÷（停損倍數 × ATR），上限為可用現金，不融資。
- 停損：盤中最低價觸及停損價即出場；跳空開低則以開盤價出場。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from statistics import NormalDist

import numpy as np
import pandas as pd

FEE, TAX = 0.001425, 0.001


@dataclass
class Costs:
    slip: float = 0.0005

    def buy(self, px: float) -> float:   # 每股實際成本
        return px * (1 + self.slip) * (1 + FEE)

    def sell(self, px: float) -> float:  # 每股實際所得
        return px * (1 - self.slip) * (1 - FEE - TAX)


@dataclass
class Result:
    equity: pd.Series
    trades: list = field(default_factory=list)  # dict(ticker, entry, exit, ret, reason)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def run(frames: dict[str, pd.DataFrame], target: pd.Series, capital: float, *,
        stop_mult: float = 2.0, risk: float = 0.02, delay: int = 1, costs: Costs | None = None) -> Result:
    """target：每個交易日收盤後想持有的代號（或 None）。delay=1 表示下一根開盤成交，2 表示晚一根。"""
    costs = costs or Costs()
    idx = target.index
    want = target.shift(delay).reindex(idx)
    atrs = {s: atr(df) for s, df in frames.items()}
    cash, held, shares, stop, entry_cost, entry_day = capital, None, 0, 0.0, 0.0, None
    blocked = None  # 停損後，同一個代號需等目標改變才能再進場
    eq, trades = [], []

    def close_pos(day, px, reason):
        nonlocal cash, held, shares
        proceeds = shares * costs.sell(px)
        trades.append(dict(ticker=held, entry=entry_day, exit=day, ret=proceeds / entry_cost - 1, reason=reason))
        cash += proceeds
        held, shares = None, 0

    prev_want = None
    for i, day in enumerate(idx):
        w = want.iloc[i] if isinstance(want.iloc[i], str) else None
        if w != prev_want:
            blocked = None
        prev_want = w
        # 1) 開盤：依前一日訊號調整部位
        if held is not None and w != held and day in frames[held].index:
            close_pos(day, frames[held].at[day, "open"], "signal")
        if held is None and w is not None and w != blocked and day in frames[w].index:
            df = frames[w]
            j = df.index.get_loc(day)
            a = atrs[w].iloc[j - 1] if j > 0 else np.nan
            if np.isfinite(a) and a > 0:
                px = df.at[day, "open"]
                equity_now = cash
                n = int(min(risk * equity_now / (stop_mult * a), cash / costs.buy(px)))
                if n > 0:
                    held, shares, entry_day = w, n, day
                    entry_cost = n * costs.buy(px)
                    cash -= entry_cost
                    stop = px - stop_mult * a
        # 2) 盤中：停損
        if held is not None and day in frames[held].index:
            bar = frames[held].loc[day]
            if bar["low"] <= stop:
                close_pos(day, min(bar["open"], stop), "stop")
                blocked = w
        # 3) 收盤：市值
        mv = 0.0
        if held is not None:
            df = frames[held]
            mv = shares * df["close"].asof(day)
        eq.append(cash + mv)
    return Result(pd.Series(eq, index=idx), trades)


# ---------- 績效指標 ----------
def stats(equity: pd.Series, trades: list) -> dict:
    r = equity.pct_change().dropna()
    yrs = max(len(r) / 252, 1e-9)
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else 0.0
    dd = (equity / equity.cummax() - 1).min()
    rets = np.array([t["ret"] for t in trades]) if trades else np.array([])
    return dict(
        cagr=(equity.iloc[-1] / equity.iloc[0]) ** (1 / yrs) - 1,
        sharpe=sharpe, max_dd=dd, trades=len(rets),
        avg_trade=rets.mean() if len(rets) else 0.0,
        win_rate=(rets > 0).mean() if len(rets) else 0.0,
        total=equity.iloc[-1] / equity.iloc[0] - 1,
    )


def deflated_sharpe(daily: pd.Series, trial_srs: list[float]) -> float:
    """Bailey & López de Prado 收縮夏普：考慮試過多少個想法後，夏普 > 0 的機率。"""
    r = daily.dropna()
    if len(r) < 30 or r.std() == 0:
        return 0.0
    sr = r.mean() / r.std()
    n = max(len(trial_srs), 2)
    v = np.var(trial_srs, ddof=1) if len(trial_srs) > 1 else 0.0
    nd, g = NormalDist(), 0.5772156649
    sr0 = np.sqrt(v) * ((1 - g) * nd.inv_cdf(1 - 1 / n) + g * nd.inv_cdf(1 - 1 / (n * np.e)))
    skew, kurt = r.skew(), r.kurt() + 3
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    return nd.cdf((sr - sr0) * np.sqrt(len(r) - 1) / denom)


def monte_carlo_dd(daily: pd.Series, runs: int = 1000, block: int = 20, seed: int = 7) -> tuple[float, float]:
    """區塊自助抽樣 1000 次：回傳（95 百分位回撤, 最差回撤）。"""
    r = daily.dropna().to_numpy()
    if len(r) < block * 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    nb = len(r) // block + 1
    dds = []
    for _ in range(runs):
        starts = rng.integers(0, len(r) - block, nb)
        path = np.concatenate([r[s:s + block] for s in starts])[: len(r)]
        eq = np.cumprod(1 + path)
        dds.append((eq / np.maximum.accumulate(eq) - 1).min())
    dds = np.array(dds)
    return float(np.percentile(dds, 5)), float(dds.min())
