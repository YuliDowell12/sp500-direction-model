import pandas as pd
from sklearn.base import clone

from model import make_predictions, make_probability_predictions


def backtest(data: pd.DataFrame, model, predictors, start=2500, step=250, use_probability=False):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[:i].copy()
        test = data.iloc[i : i + step].copy()

        model_fold = clone(model)
        model_fold.fit(train[predictors], train["Target"])

        if use_probability:
            preds = make_probability_predictions(model_fold, test, predictors)
        else:
            preds = make_predictions(model_fold, test, predictors)

        combined = pd.concat([test["Target"], preds], axis=1)
        all_predictions.append(combined)

    return pd.concat(all_predictions)


def evaluate(predictions: pd.DataFrame):
    from sklearn.metrics import precision_score

    score = precision_score(predictions["Target"], predictions["Predictions"])
    return score
