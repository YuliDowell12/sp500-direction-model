import pandas as pd

HORIZONS = [2, 5, 60, 250, 1000]


def add_technical_features(data: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    """Add rolling close ratios and lagged trend features."""
    data = data.copy()

    for horizon in horizons:
        ratio_column = f"Close_Ratio_{horizon}"
        rolling_mean = data["Close"].rolling(horizon).mean()
        data[ratio_column] = data["Close"] / rolling_mean

        trend_column = f"Trend_{horizon}"
        data[trend_column] = data["Target"].shift(1).rolling(horizon).sum()

    return data.dropna().copy()
