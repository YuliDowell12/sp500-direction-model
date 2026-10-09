from data import clean_data, download_sp500_history
from features import HORIZONS, add_technical_features
from model import fit_model, make_predictions
from backtest import backtest, evaluate

PREDICTORS = ["Close", "Volume", "Open", "High", "Low"]


def main():
    sp500 = download_sp500_history()
    sp500 = clean_data(sp500)

    baseline_model = fit_model(sp500.iloc[:-100], PREDICTORS, n_estimators=100, min_samples_split=100)
    basic_preds = make_predictions(baseline_model, sp500.iloc[-100:], PREDICTORS)
    basic_score = evaluate(basic_preds.to_frame().assign(Target=sp500.iloc[-100:]["Target"]))
    print(f"Basic model precision: {basic_score:.4f}")

    feature_data = add_technical_features(sp500)
    engineered_predictors = PREDICTORS + [f"Close_Ratio_{h}" for h in HORIZONS] + [f"Trend_{h}" for h in HORIZONS]
    engineered_model = fit_model(feature_data.iloc[:-100], engineered_predictors, n_estimators=200, min_samples_split=50)
    predictions = backtest(feature_data, engineered_model, engineered_predictors, use_probability=True)
    backtest_score = evaluate(predictions)
    print(f"Backtest precision: {backtest_score:.4f}")


if __name__ == "__main__":
    main()
