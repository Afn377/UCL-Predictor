from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "src" / "data" / "processed" / "model_dataset.csv"

CLASSES = [0,1,2]
TEMPORAL_SPLIT_DATE = "2024-07-01" # 2024 UCL started July 9, 2024

ELO_FEATURES = ["elo_diff"]
FULL_FEATURES = ["elo_diff", "ppg_5_diff", "goal_difference_5_diff"]


def temporal_train_test_split(df):
    train = df[df["date"] < TEMPORAL_SPLIT_DATE].copy()
    test = df[df["date"] >= TEMPORAL_SPLIT_DATE].copy()
    return train, test



def naive_base_rate_probabilities(train, n_rows):
    # for this just count the number of times each result occurs in training set then normalize it to a value 
    # between 0 and 1 and order by the CLASSES list then return the probabilities as a dataframe w n_rows
    class_probabilities = train["result"].value_counts(normalize=True).reindex(CLASSES, fill_value=0)
    return pd.DataFrame([class_probabilities.to_dict()] * n_rows)


def train_logistic_model(train, features):
    model = LogisticRegression(max_iter=1000)
    model.fit(train[features], train["result"])
    return model


def predict_logistic_probabilities(model, data, features):
    probabilities = model.predict_proba(data[features])
    return probabilities