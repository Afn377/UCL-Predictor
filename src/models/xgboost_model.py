from __future__ import annotations

import os

import pandas as pd
from xgboost import XGBClassifier


os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


def train_xgboost_classifier(
    train: pd.DataFrame,
    features: list[str],
    random_seed: int = 7,
) -> XGBClassifier:
    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        n_estimators=200,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=10,
        reg_lambda=2.0,
        random_state=random_seed,
        n_jobs=1,
    )
    model.fit(train[features], train["result"])
    return model
