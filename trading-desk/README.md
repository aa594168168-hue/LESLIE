# 台股 ETF 交易台

一人對沖基金：6 個 Agent、一份規則手冊、模擬盤先行。

| 檔案 | 用途 |
| --- | --- |
| `DESK_RULES.md` | 交易台規則，每個 Agent 行動前必讀 |
| `TEAM_AND_SCHEDULE.md` | 團隊、排程、研究計畫 |
| `research/run_research.py` | 夜間研究：5 個假設的回測、樣本外、滾動前進、蒙地卡羅、評分 |
| `research/reports/` | 每晚的研究報告（`.md` 給人看，`.json` 給程式讀） |
| `paper/paper.py` | 模擬交易機器人（策略 A），自建帳本 `state.json`、`ledger.csv` |
| `paper/morning.py` | 晨報資料 |
| `tradingview/strategy_A.pine` | TradingView 指標與提醒 |
| `journal/` | 交易日誌：標的、想法、證據、下一步 |

## 不會寫程式也能照做的部署步驟

1. **資料金鑰**：到 finmindtrade.com 註冊免費帳號，把 token 存到雲端環境的網路金鑰（api.finmindtrade.com）。程式不放金鑰。
2. **每晚收盤處理**：`python3 trading-desk/paper/paper.py run`（需安裝 pandas、numpy）
3. **看帳戶**：`python3 trading-desk/paper/paper.py status`
4. **重跑研究**：`cd trading-desk/research && python3 run_research.py`
5. **TradingView 提醒**：打開 TradingView → 圖表下方「Pine 編輯器」→ 貼上 `tradingview/strategy_A.pine` 全部內容 → 「加到圖表」→ 右上角「提醒」→ 條件選「Leslie 策略A」→「策略A 進場」或「策略A 出場」→ 建立。對每檔 ETF（如 TWSE:0050）各做一次。

## 為什麼不是 Alpaca

Alpaca 只支援美股與加密貨幣，不支援台股。實盤需要台灣券商 API（例如永豐金 Shioaji、富果），接上前必須經使用者批准。
