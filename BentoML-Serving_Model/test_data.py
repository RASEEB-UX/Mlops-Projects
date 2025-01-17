import pandas as pd

from data_ingest import get_data
import datetime as dt
import yfinance as yf
yf.pdr_override()

df = get_data('SPY', '2025-01-01', dt.date.today())
df.head(5)

df.to_csv('SPY_data.csv', index=False)