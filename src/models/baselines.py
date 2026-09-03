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
    # predict probabilities for each of 0, 1, 2 classes and return the result
    probabilities = model.predict_proba(data[features])
    return probabilities


def evaluate_probabilities(y_true, probabilities):
    return {
        "log_loss": log_loss(y_true, probabilities, labels=CLASSES),
        # pick the class with highest probability to compare to the true outcome
        "accuracy": accuracy_score(y_true, probabilities.argmax(axis=1))
    }


def run_baselines(input_path=DEFAULT_INPUT):
    # tell pandas to treat the date column as a datetime object
    df = pd.read_csv(input_path, parse_dates=["date"])
    train, test = temporal_train_test_split(df)

    rows = []

    # predict the base rate probabilities for each class
    naive_probabilities = naive_base_rate_probabilities(train, len(test)).to_numpy()
    rows.append({"model":"naive_base_rate", **evaluate_probabilities(test["result"], naive_probabilities)})

    # predict the probabilities using logistic regression with only ELO
    elo_model = train_logistic_model(train, ELO_FEATURES)
    elo_probabilities = predict_logistic_probabilities(elo_model, test, ELO_FEATURES)
    rows.append({"model":"elo_logistic", **evaluate_probabilities(test["result"], elo_probabilities)})

    # predict probabilities using logistic regression with all features
    feature_model = train_logistic_model(train, FULL_FEATURES)
    feature_probabilities = predict_logistic_probabilities(feature_model, test, FULL_FEATURES)
    rows.append({"model":"feature_logistic", **evaluate_probabilities(test["result"], feature_probabilities)})

    return pd.DataFrame(rows)


def main():
    results = run_baselines()
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()