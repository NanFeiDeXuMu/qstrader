import os
import pandas as pd
import yfinance as yf

SYMBOLS = ['SPY', 'AGG', 'GLD', 'SHY', 'TLT']
START = '2003-01-01'
END   = '2024-01-01'

def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(-1)
        data.columns.name = None

    # QSTrader CSVDailyBarDataSource expects: Date,Open,High,Low,Close,Volume
    out = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

    out_path = os.path.join(os.path.dirname(__file__), f'{ticker}.csv')
    out.to_csv(out_path, index_label='Date')
    print(f"[OK] {ticker}: {len(out)} rows saved to {out_path}")


if __name__ == '__main__':
    for sym in SYMBOLS:
        get_data(sym, START, END)
