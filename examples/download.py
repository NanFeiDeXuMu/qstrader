import pandas as pd
import yfinance as yf

list_of_tickers = ["AGG", "SPY", "GLD"]
for ticker in list_of_tickers:
    data = yf.download(ticker, start="2003-09-30", end="2019-12-31")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(-1)
        data.columns.name = None

    data.to_csv(f"{ticker}.csv", index_label="Date")