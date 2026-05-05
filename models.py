from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


@dataclass
class TrainingResult:
    algorithm: str
    model: Any
    training_time_sec: float
    metrics: Dict[str, float]
    y_test: np.ndarray | None = None
    y_pred: np.ndarray | None = None
    confusion: np.ndarray | None = None
    extra: Dict[str, Any] | None = None


class ModelFactory:

    SUPPORTED = [
        "Linear Regression",
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "K-Means Clustering",
        "Support Vector Machine (SVM)",
    ]

    def _create_model(self, algorithm: str, params: Dict[str, Any]) -> Any:
        if algorithm == "Linear Regression":
            return LinearRegression()

        if algorithm == "Logistic Regression":
            return LogisticRegression(
                C=float(params.get("C", 1.0)),
                max_iter=int(params.get("max_iter", 300)),
                solver=params.get("solver", "lbfgs"),
                class_weight="balanced"
            )

        if algorithm == "Decision Tree":
            return DecisionTreeClassifier(
                max_depth=int(params.get("max_depth", 5)),
                min_samples_split=int(params.get("min_samples_split", 2)),
                random_state=42,
                class_weight="balanced"
            )

        if algorithm == "Random Forest":
            return RandomForestClassifier(
                n_estimators=int(params.get("n_estimators", 100)),
                max_depth=int(params.get("max_depth", 8)),
                min_samples_split=int(params.get("min_samples_split", 2)),
                random_state=42,
                class_weight="balanced"
            )

        if algorithm == "Support Vector Machine (SVM)":
            return SVC(
                C=float(params.get("C", 1.0)),
                kernel=params.get("kernel", "rbf"),
                gamma=params.get("gamma", "scale"),
                class_weight="balanced"
            )

        if algorithm == "K-Means Clustering":
            return KMeans(
                n_clusters=int(params.get("n_clusters", 3)),
                n_init=10,
                random_state=42,
            )

        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # 🔥 FIXED + IMPROVED
    def train_supervised(
        self,
        algorithm: str,
        params: Dict[str, Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> TrainingResult:

        model = self._create_model(algorithm, params)

        # 🔥 Hyperparameter tuning
        param_grid = {}
        if algorithm == "Logistic Regression":
            param_grid = {"C": [0.1, 1, 10]}
        elif algorithm == "Decision Tree":
            param_grid = {"max_depth": [3, 5, 10]}
        elif algorithm == "Random Forest":
            param_grid = {"n_estimators": [50, 100], "max_depth": [5, 10]}
        elif algorithm == "Support Vector Machine (SVM)":
            param_grid = {"C": [0.1, 1, 10]}

        start = time.perf_counter()

        if param_grid:
            grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)

        training_time = time.perf_counter() - start
        y_pred = model.predict(X_test)

        # 🔥 Cross Validation Score
        cv_score = cross_val_score(model, X_train, y_train, cv=5).mean()

        # 🔥 Regression
        if algorithm == "Linear Regression":
            metrics = {
                "MSE": float(mean_squared_error(y_test, y_pred)),
                "R2": float(r2_score(y_test, y_pred)),
                "CV Score": float(cv_score),
            }

            return TrainingResult(
                algorithm=algorithm,
                model=model,
                training_time_sec=training_time,
                metrics=metrics,
                y_test=y_test,
                y_pred=y_pred,
            )

        # 🔥 Classification
        metrics = {
            "Accuracy": float(accuracy_score(y_test, y_pred)),
            "Precision": float(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "Recall": float(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "F1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
            "CV Score": float(cv_score),
        }

        conf = confusion_matrix(y_test, y_pred)

        return TrainingResult(
            algorithm=algorithm,
            model=model,
            training_time_sec=training_time,
            metrics=metrics,
            y_test=y_test,
            y_pred=y_pred,
            confusion=conf,
        )

    # 🔥 KMeans (fixed indentation)
    def train_kmeans(
        self, params: Dict[str, Any], X: np.ndarray
    ) -> Tuple[TrainingResult, np.ndarray]:

        algorithm = "K-Means Clustering"
        model = self._create_model(algorithm, params)

        start = time.perf_counter()
        clusters = model.fit_predict(X)
        training_time = time.perf_counter() - start

        metrics = {"Inertia": float(model.inertia_)}

        result = TrainingResult(
            algorithm=algorithm,
            model=model,
            training_time_sec=training_time,
            metrics=metrics,
            extra={"clusters": clusters},
        )

        return result, clusters