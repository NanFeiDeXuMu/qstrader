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