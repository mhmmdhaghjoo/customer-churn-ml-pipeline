"""Model definitions. Preprocessing lives inside each pipeline, so it is
fitted on training data only and travels with the saved model."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

RECENCY_ONLY = ["recency_days"]


def _preprocessor(scale: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def candidate_models(seed: int = 42) -> dict[str, Pipeline]:
    """The models that are compared, from simplest to most flexible."""
    return {
        # Baseline: "the longer since the last order, the more likely gone".
        # Any real model has to beat this to justify its complexity.
        "recency_only": Pipeline(
            [
                ("select", ColumnTransformer([("recency", "passthrough", RECENCY_ONLY)])),
                ("scale", StandardScaler()),
                ("model", LogisticRegression()),
            ]
        ),
        "logistic_regression": Pipeline(
            [
                ("prep", _preprocessor(scale=True)),
                ("model", LogisticRegression(max_iter=2000, C=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", _preprocessor(scale=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400, min_samples_leaf=10, n_jobs=-1, random_state=seed
                    ),
                ),
            ]
        ),
        "gradient_boosting": Pipeline(
            [
                ("prep", _preprocessor(scale=False)),
                ("to_dense", FunctionTransformer(_to_dense, accept_sparse=True)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=3, learning_rate=0.05, max_iter=300, random_state=seed
                    ),
                ),
            ]
        ),
    }


def _to_dense(matrix):  # noqa: ANN001, ANN202 - scipy sparse or ndarray in, ndarray out
    return matrix.toarray() if hasattr(matrix, "toarray") else matrix
