import numpy as np
import pandas as pd


from src.models.baselines import (
    CLASSES,
    ELO_FEATURES,
    evaluate_probabilities,
    naive_base_rate_probabilities,
    temporal_train_test_split,
    train_logistic_model,
)


def make_dataset():
    rows=[]
    for i in range(67):
        rows.append({
            "date": pd.Timestamp("2022-07-01") + pd.Timedelta(days=i),
            "elo_diff": i - 33,
            "ppg_5_diff": (i%5)-3,
            "goal_difference_5_diff": (i%5)-2,
            "result": i%3
        })

    for i in range(15):
        rows.append({
            "date": pd.Timestamp("2024-07-01") + pd.Timedelta(days=i),
            "elo_diff": i,
            "ppg_5_diff": i%5,
            "goal_difference_5_diff": i%4,
            "result": i%3
        })

def test_temporal_split_overlap():
    # make sure the temporal split does not overlap in dates between train and test sets
    df = make_dataset()
    train, test = temporal_train_test_split(df)
    assert train["date"].max() < pd.Timestamp("2024-07-01")
    assert test["date"].min() >= pd.Timestamp("2024-07-01")


def test_naive_base_rate_probabilities_sum_one():

    # make sure all rows in the probabilities dataframe sum to 1
    df = make_dataset()
    train, test = temporal_train_test_split(df)
    probabilities = naive_base_rate_probabilities(train, len(test))
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_logistic_model_returns_three_probs():
    df = make_dataset()
    train, test = temporal_train_test_split(df)
    model = train_logistic_model(train, ELO_FEATURES)
    probabilities = model.predict_proba(test[ELO_FEATURES])
    assert probabilities.shape == (len(test), 3)
    assert np.allclose(probabilities.sum(axis=1), 1)


def test_evaluate_probabilities_returns_dict():
    y_true = np.array([0,1,2])
    probs = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
        ]
    )

    result = evaluate_probabilities(y_true, probs)
    assert set(result) == {"accuracy", "log_loss"}
    assert result["accuracy"] == 1.0
    assert result["log_loss"] > 0