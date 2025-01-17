import pandas as pd
import yfinance as yf
import datetime as dt

def get_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    data.reset_index(inplace=True)
    data.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    data['Date'] = pd.to_datetime(data['Date'])
    return data

# get_data('SPY', '1990-01-01', dt.date.today())
