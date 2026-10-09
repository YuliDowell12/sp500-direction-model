# S&P 500 Direction Model

A machine learning project that predicts whether the S&P 500 will close higher tomorrow than today. It uses a random forest classifier on engineered price features and is evaluated with walk-forward backtesting.

This is a modeling and data-handling exercise, not a trading strategy.

## How it works

1. **Data (`data.py`)**: downloads the full S&P 500 price history (ticker `^GSPC`) from Yahoo Finance using `yfinance`, trims it to 1990 onward, and creates the label: 1 if tomorrow's close is higher than today's, otherwise 0.
2. **Features (`features.py`)**: for windows of 2, 5, 60, 250, and 1000 days, builds
   - a close ratio (today's close divided by its rolling average), and
   - a trend count (how many of the previous N days were up days).

   Trend features use only past days, so the model never sees information that wouldn't have been known at prediction time (no data leakage).
3. **Model (`model.py`)**: a scikit-learn `RandomForestClassifier`. It can predict a plain 0/1, or only call "up" when its predicted probability is at least 60%.
4. **Backtest (`backtest.py`)**: walk-forward validation. The model trains on all data before a cutoff, tests on the next 250 trading days, then the cutoff moves forward and it repeats. Rows are never shuffled, since random splits would let the model train on days after the ones it's tested on.
5. **Runner (`main.py`)**: compares a baseline model (raw prices only) against the engineered-feature model and prints precision for each.

## Results

| Model | Precision |
|---|---|
| Baseline (raw prices, last 100 days) | 0.0000 |
| Engineered features (walk-forward backtest) | 0.5741 |

Precision is the share of predicted "up" days that really were up. The baseline scored 0.0 because it never predicted "up" on the last 100 days, which leaves precision undefined (scikit-learn reports 0.0 and a warning). Raw price levels keep drifting upward, so a model trained on them struggles to generalize to new price ranges. Ratio-based features stay in a stable range.

Precision should be read against the base rate: the market rises on a bit more than half of trading days, so a model that always guesses "up" would already score above 0.5. Results will vary slightly as new trading days are added.

## Run it

Requires Python 3 and an internet connection (the data is downloaded each run).

```
python3 -m pip install -r requirements.txt
python3 main.py
```

The backtest retrains the model about 20 times, so the full run takes a minute or two.

## Limitations

- Precision alone ignores missed opportunities, trading costs, and the size of gains and losses, so a good score here doesn't mean a profitable strategy.
- Only one ticker, with no transaction costs or position sizing.
- Known issue: `clean_data` sets the final row's label to 0 because tomorrow's close isn't known yet. This slightly affects the baseline test on the most recent 100 days.

## Tech

Python, pandas, scikit-learn, yfinance
