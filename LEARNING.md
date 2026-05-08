```txt
git restore fileName
```

modified之后,add之前,回退到上一次commit的时候的版本.

![image-20260507195254142](assets/image-20260507195254142.png)

第一个金融分析图.有两个要注意:

```python
import pandas as pd
import yfinance as yf

def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(-1)
        data.columns.name = None

    if(data.to_csv(f"{ticker}.csv")):
        print(f"Data for {ticker} saved to {ticker}.csv")
    else:
        print(f"Failed to save data for {ticker}")
```

`isinstance`函数指定`columns`为对象,不要写成`data`了. 抽象化后成为可以`import`的工具函数, 在每一个`examples`中可以灵活调用而不必硬编码或者反复下载. 

`columns`只有`Index`或者`MultiIndex`两种形态. `.columns.droplevel`指定抛弃某一层的索引,这里把聚合类名称扔掉.

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

```python
import download
download.get_data(ticker, start, end)
```

对于自己写的脚本,直接import就可以使用了.