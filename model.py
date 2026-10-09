import pandas as pd
from sklearn.ensemble import RandomForestClassifier

PROBABILITY_THRESHOLD = 0.6


def fit_model(train: pd.DataFrame, predictors, n_estimators=200, min_samples_split=50, random_state=1):
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )
    model.fit(train[predictors], train["Target"])
    return model


def make_predictions(model, test: pd.DataFrame, predictors):
    preds = model.predict(test[predictors])
    return pd.Series(preds, index=test.index, name="Predictions")


def make_probability_predictions(model, test: pd.DataFrame, predictors, threshold=PROBABILITY_THRESHOLD):
    probabilities = model.predict_proba(test[predictors])[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    return pd.Series(predictions, index=test.index, name="Predictions")
