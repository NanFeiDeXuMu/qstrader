```txt
git restore fileName
```

modified之后,add之前,回退到上一次commit的时候的版本.

![image-20260507195254142](assets/image-20260507195254142.png)

第一个金融分析图.有两个要注意:

```python
import pandas as pd
import yfinance as yf

list_of_tickers = ["AGG", "SPY", "GLD"]
for ticker in list_of_tickers:
    data = yf.download(ticker, start="2003-09-30", end="2019-12-31")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(-1)
        data.columns.name = None

    data.to_csv(f"{ticker}.csv", index_label="Date")
```

数据清理时可能遇见多层表头,此时直接`.columns.droplevel(-1)`将`ticker`去除.避免出现格式混乱无法proceed的情况.

```python
data_source = CSVDailyBarDataSource(csv_dir, Equity, csv_symbols=strategy_symbols, adjust_prices=False)
```

可见`CSVDailyBarDataSource`中的定义

```python
def __init__(self, csv_dir, asset_type, adjust_prices=True, csv_symbols=None):
        self.csv_dir = csv_dir
        self.asset_type = asset_type
        self.adjust_prices = adjust_prices
        self.csv_symbols = csv_symbols

        self.asset_bar_frames = self._load_csvs_into_dfs()
        self.asset_bid_ask_frames = self._convert_bars_into_bid_ask_dfs()
```

默认`adjusted_prices=True`,但注意yahoo通常没有`adjusted_price`这一栏,因此应该disable掉.