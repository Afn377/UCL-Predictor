"""Shared logistic regression fitting for the two football feature sets."""

from dataclasses import dataclass

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class FeatureModel:
    model: object
    features: list[str]

    def predict_proba(self, data):
        # pin feature names with the estimator so columns can't drift
        return self.model.predict_proba(data[self.features])


def train_feature_model(train, features, regularization=0.1):
    # fit medians/scale on train only; indicators flag absent xG
    model = make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(),
        LogisticRegression(C=regularization, max_iter=3000),
    )
    model.fit(train[features], train.result)
    return FeatureModel(model, list(features))
