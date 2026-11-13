"""夜間研究：回測 5 個假設 → 樣本內／外 → 滾動前進 → 蒙地卡羅 → 壓力測試 → 評分 → 報告。

用法：python3 run_research.py [--date YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

import data
import engine
from strategies import STRATEGIES, THESIS

CAPITAL = 200_000
OOS_START = pd.Timestamp("2024-01-01")
WF_YEARS = [2022, 2023, 2024, 2025, 2026]
HERE = Path(__file__).parent


def backtest(cfg, frames, cal, value, ex_dates, slip=0.0005, delay=1):
    kw = {cfg["param"]: value}
    if cfg["fn"].__name__ == "pre_dividend":
        kw["ex_dates"] = ex_dates
    targets = cfg["fn"](frames, cal, **kw)
    cap = CAPITAL / len(targets)
    eq, trades = None, []
    for tgt in targets.values():
        r = engine.run(frames, tgt, cap, stop_mult=cfg["stop"], delay=delay, costs=engine.Costs(slip))
        eq = r.equity if eq is None else eq + r.equity
        trades += r.trades
    return eq, trades


def slice_stats(eq, trades, start=None, end=None):
    e = eq[(eq.index >= start) if start is not None else slice(None)]
    if end is not None:
        e = e[e.index <= end]
    t = [x for x in trades if (start is None or x["entry"] >= start) and (end is None or x["entry"] <= end)]
    return engine.stats(e, t)


def score(oos, dsr, dd, trades, edge):
    def band(x, cuts):  # cuts 由低到高，回傳 1–5
        return 1 + sum(x >= c for c in cuts)
    return dict(
        oos_sharpe=band(oos, [0.0, 0.5, 1.0, 1.5]),
        dsr=band(dsr, [0.5, 0.8, 0.9, 0.95]),
        max_dd=band(dd, [-0.30, -0.20, -0.15, -0.10]),
        trades=band(trades, [30, 50, 100, 200]),
        edge=band(edge, [0.0, 0.001, 0.003, 0.006]),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default=date.today().isoformat())
    args = ap.parse_args()

    frames = {}
    missing = []
    for s in data.UNIVERSE:
        try:
            frames[s] = data.load(s)
        except Exception as e:  # 額度用完或資料缺漏：記下來、照常跑可用的
            missing.append(f"{s}（{e}）")
    cal = frames["0050"].index
    ex_dates = {s: [pd.Timestamp(d) for d, _, split in df.attrs["events"] if not split] for s, df in frames.items()}

    # 1) 所有參數組合（多重檢驗的試驗數）
    runs = {}
    for name, cfg in STRATEGIES.items():
        for v in cfg["grid"]:
            runs[(name, v)] = backtest(cfg, frames, cal, v, ex_dates)
    trial_srs = []
    for eq, _ in runs.values():
        r = eq.pct_change().dropna()
        trial_srs.append(r.mean() / r.std() if r.std() > 0 else 0.0)

    bh = frames["0050"]["close"]
    bench = {
        "is": engine.stats(bh[bh.index < OOS_START], []),
        "oos": engine.stats(bh[bh.index >= OOS_START], []),
    }

    results = {}
    for name, cfg in STRATEGIES.items():
        eq, trades = runs[(name, cfg["base"])]
        ins = slice_stats(eq, trades, end=OOS_START - pd.Timedelta(days=1))
        oos = slice_stats(eq, trades, start=OOS_START)
        daily = eq.pct_change()
        dsr = engine.deflated_sharpe(daily, trial_srs)
        mc95, mcworst = engine.monte_carlo_dd(daily)

        # 滾動前進：每年用之前的資料挑參數，套用到該年
        wf_parts, wf_pick = [], {}
        for y in WF_YEARS:
            ys, ye = pd.Timestamp(f"{y}-01-01"), pd.Timestamp(f"{y}-12-31")
            best = max(cfg["grid"], key=lambda v: engine.stats(runs[(name, v)][0][runs[(name, v)][0].index < ys], [])["sharpe"])
            wf_pick[y] = best
            e = runs[(name, best)][0]
            wf_parts.append(e.pct_change()[(e.index >= ys) & (e.index <= ye)])
        wf = pd.concat(wf_parts).dropna()
        wf_sharpe = wf.mean() / wf.std() * np.sqrt(252) if wf.std() > 0 else 0.0

        # 壓力：滑點加倍、晚一根 K 線
        eq2, tr2 = backtest(cfg, frames, cal, cfg["base"], ex_dates, slip=0.001)
        eq3, tr3 = backtest(cfg, frames, cal, cfg["base"], ex_dates, delay=2)
        stress = {
            "slip_x2_oos": slice_stats(eq2, tr2, start=OOS_START),
            "delay_1bar_oos": slice_stats(eq3, tr3, start=OOS_START),
            "y2020": slice_stats(eq, trades, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-12-31")),
            "y2022": slice_stats(eq, trades, pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
        }
        sc = score(oos["sharpe"], dsr, min(oos["max_dd"], ins["max_dd"]), oos["trades"], oos["avg_trade"])
        total = sum(sc.values())
        profitable = oos["trades"] >= 30 and oos["total"] > 0
        results[name] = dict(base=cfg["base"], param=cfg["param"], ins=ins, oos=oos, dsr=dsr,
                             mc95=mc95, mcworst=mcworst, wf_sharpe=wf_sharpe, wf_pick=wf_pick,
                             stress=stress, score=sc, total=total, profitable=profitable,
                             passed=profitable and total >= 18)

    out_json = HERE / "reports" / f"{args.date}-research.json"
    out_json.write_text(json.dumps(dict(results=results, bench=bench, missing=missing, n_trials=len(trial_srs),
                                        universe=list(frames), data_end=str(cal[-1].date())),
                                   default=lambda o: bool(o) if isinstance(o, np.bool_) else float(o) if isinstance(o, (np.floating, np.integer)) else str(o),
                                   ensure_ascii=False, indent=1))
    write_report(args.date, results, bench, missing, len(trial_srs), list(frames), cal)


def pct(x):
    return f"{x * 100:+.1f}%"


def write_report(day, results, bench, missing, n_trials, universe, cal):
    rank = sorted(results.items(), key=lambda kv: (kv[1]["passed"], kv[1]["total"], kv[1]["oos"]["sharpe"]), reverse=True)
    L = [f"# 夜間研究報告 {day}", "",
         f"- 資料：FinMind 台股日線（已還原除息與分割），{cal[0].date()} 至 {cal[-1].date()}",
         f"- 標的：{'、'.join(universe)}" + (f"；**缺資料**：{'；'.join(missing)}" if missing else ""),
         f"- 樣本內：至 2023-12-31；樣本外：2024-01-01 起。共試了 {n_trials} 個參數組合（收縮夏普已依此扣分）",
         "- 成本：手續費 0.1425%×2、證交稅 0.1%、滑點每邊 0.05%；訊號收盤產生、下一根開盤成交",
         f"- 基準 0050 買進持有：樣本內夏普 {bench['is']['sharpe']:.2f}、樣本外夏普 {bench['oos']['sharpe']:.2f}、"
         f"樣本外報酬 {pct(bench['oos']['total'])}、樣本外最大回撤 {pct(bench['oos']['max_dd'])}", "",
         "## 排名", "",
         "| # | 策略 | 總分/25 | 樣本外夏普 | 收縮夏普 | 最大回撤 | 樣本外交易 | 每筆淨報酬 | 通過 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for i, (n, r) in enumerate(rank, 1):
        L.append(f"| {i} | {n} | {r['total']} | {r['oos']['sharpe']:.2f} ({r['score']['oos_sharpe']}) | "
                 f"{r['dsr']:.2f} ({r['score']['dsr']}) | {pct(min(r['oos']['max_dd'], r['ins']['max_dd']))} ({r['score']['max_dd']}) | "
                 f"{r['oos']['trades']} ({r['score']['trades']}) | {pct(r['oos']['avg_trade'])} ({r['score']['edge']}) | "
                 f"{'✅' if r['passed'] else '❌'} |")
    L += ["", "括號內為 1–5 分。通過＝樣本外 ≥ 30 筆、扣費後報酬為正、總分 ≥ 18。", "",
          "## 與買進持有 0050 的比較（風控官註記）", "",
          f"樣本外期間 0050 買進持有報酬 {pct(bench['oos']['total'])}、夏普 {bench['oos']['sharpe']:.2f}。"
          "下表為各策略同期報酬：策略的優勢在於**回撤較小、風險調整後報酬**，絕對報酬可能遠低於大盤。", "",
          "| 策略 | 樣本外報酬 | 樣本外夏普 | 樣本外最大回撤 |", "| --- | --- | --- | --- |"]
    for n, r in rank:
        L.append(f"| {n} | {pct(r['oos']['total'])} | {r['oos']['sharpe']:.2f} | {pct(r['oos']['max_dd'])} |")
    L.append(f"| 0050 買進持有 | {pct(bench['oos']['total'])} | {bench['oos']['sharpe']:.2f} | {pct(bench['oos']['max_dd'])} |")
    L.append("")
    for i, (n, r) in enumerate(rank[:3], 1):
        thesis, cp, fail = THESIS[n]
        s = r["stress"]
        L += [f"## 第 {i} 名：{n}", "",
              f"- **論點**：{thesis}",
              f"- **對手盤**：{cp}",
              f"- **參數**：{r['param']} = {r['base']}（滾動前進每年挑選：{r['wf_pick']}）",
              "- **倉位**：單筆風險 2% ÷（停損倍數 × ATR14），不超過可用現金",
              f"- **樣本內 vs 樣本外**：夏普 {r['ins']['sharpe']:.2f} → {r['oos']['sharpe']:.2f}；"
              f"年化 {pct(r['ins']['cagr'])} → {pct(r['oos']['cagr'])}；交易 {r['ins']['trades']} → {r['oos']['trades']} 筆；"
              f"勝率 {r['oos']['win_rate'] * 100:.0f}%",
              f"- **滾動前進夏普**（2022–2026）：{r['wf_sharpe']:.2f}",
              f"- **蒙地卡羅 1000 次回撤**：95% 情境不超過 {pct(r['mc95'])}，最差 {pct(r['mcworst'])}",
              f"- **壓力測試**：滑點加倍樣本外夏普 {s['slip_x2_oos']['sharpe']:.2f}；晚一根 K 線 {s['delay_1bar_oos']['sharpe']:.2f}；"
              f"2020 年報酬 {pct(s['y2020']['total'])}；2022 年報酬 {pct(s['y2022']['total'])}",
              f"- **它為什麼會失敗**：{fail}", ""]
    L += ["## 全部否決的理由", ""]
    for n, r in rank:
        if not r["passed"]:
            why = []
            if r["oos"]["trades"] < 30:
                why.append(f"樣本外只有 {r['oos']['trades']} 筆交易")
            if r["oos"]["total"] <= 0:
                why.append("樣本外扣費後虧損")
            if r["total"] < 18:
                why.append(f"總分 {r['total']} < 18")
            L.append(f"- **{n}**：{'；'.join(why)}")
    L += ["", "> 所有數字皆為歷史回測的估算，不代表未來報酬。未計入股利所得稅與二代健保補充保費。"]
    (HERE / "reports" / f"{day}-research.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
