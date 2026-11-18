"""晨報資料：市場概況、策略排名、模擬盤狀態。高影響事件由排程中的 Claude 以網路搜尋補上。

用法：python3 trading-desk/paper/morning.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "research"))
import data  # noqa: E402

df = data.load("0050")  # 使用快取；paper.py run 已在收盤後更新
c = df["close"]
r1, r5, r20 = c.iloc[-1] / c.iloc[-2] - 1, c.iloc[-1] / c.iloc[-6] - 1, c.iloc[-1] / c.iloc[-21] - 1
vol = c.pct_change().tail(20).std() * 252 ** 0.5
print(f"資料日：{df.index[-1].date()}")
print(f"0050 收 {df['raw_close'].iloc[-1]}：日 {r1:+.1%}、週 {r5:+.1%}、月 {r20:+.1%}；"
      f"{'站上' if c.iloc[-1] > c.rolling(60).mean().iloc[-1] else '跌破'} 60 日線、"
      f"{'站上' if c.iloc[-1] > c.rolling(200).mean().iloc[-1] else '跌破'} 200 日線；20 日年化波動 {vol:.0%}")
reports = sorted((HERE.parent / "research" / "reports").glob("*-research.json"))
if reports:
    rep = json.loads(reports[-1].read_text())
    rank = sorted(rep["results"].items(), key=lambda kv: (kv[1]["passed"], kv[1]["total"]), reverse=True)[:3]
    print(f"策略排名（{reports[-1].stem}）：")
    for i, (n, x) in enumerate(rank, 1):
        print(f"  {i}. {n} {int(x['total'])}/25 樣本外夏普 {float(x['oos']['sharpe']):.2f} {'通過' if x['passed'] else '否決'}")
state = HERE / "state.json"
print("模擬盤：", state.read_text() if state.exists() else "尚未開始")
