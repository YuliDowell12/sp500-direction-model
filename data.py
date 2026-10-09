import pandas as pd
import yfinance as yf


def download_sp500_history(symbol="^GSPC", start_date="1990-01-01") -> pd.DataFrame:
    """Fetch S&P 500 history from Yahoo Finance and trim to the requested date."""
    data = yf.Ticker(symbol).history(period="max")
    return data.loc[start_date:].copy()


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Remove unused columns and create the target label for tomorrow's direction."""
    data = data.copy()
    data = data.drop(columns=["Dividends", "Stock Splits"], errors="ignore")
    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)
    return data
