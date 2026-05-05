from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler


EXAMPLES_DIR = Path("data")


def load_dataset(file) -> pd.DataFrame:
    if file is None:
        raise ValueError("No file provided.")
    df = pd.read_csv(file)
    if df.empty:
        raise ValueError("Dataset is empty.")
    return df


def get_example_datasets() -> Dict[str, Path]:
    EXAMPLES_DIR.mkdir(exist_ok=True)
    return {
        "Iris Classification": EXAMPLES_DIR / "iris_sample.csv",
        "Housing Regression": EXAMPLES_DIR / "housing_sample.csv",
    }


# 🔥 FIXED preprocessing (no dtype crash)
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for col in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[col]):
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())
        else:
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

    return cleaned


# 🔥 Better encoding (handles all non-numeric types)
def encode_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    encoded = df.copy()

    for col in columns:
        if not pd.api.types.is_numeric_dtype(encoded[col]):
            le = LabelEncoder()
            encoded[col] = le.fit_transform(encoded[col].astype(str))

    return encoded


def prepare_supervised_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    test_size: float,
    scale: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    dataset = preprocess_dataframe(df)
    dataset = encode_features(dataset, feature_cols + [target_col])

    X = dataset[feature_cols].values
    y = dataset[target_col].values

    stratify = y if len(np.unique(y)) < 20 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=stratify
    )

    # 🔥 SAFE FEATURE SELECTION (no crash)
    try:
        from sklearn.feature_selection import SelectKBest, f_classif, f_regression

        if len(feature_cols) > 3:
            if len(np.unique(y_train)) < 20:
                score_func = f_classif
            else:
                score_func = f_regression

            selector = SelectKBest(score_func=score_func, k=min(5, X_train.shape[1]))

            X_train = selector.fit_transform(X_train, y_train)
            X_test = selector.transform(X_test)

    except Exception as e:
        print("Feature selection skipped:", e)

    # 🔥 Scaling
    if scale:
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def prepare_clustering_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    scale: bool = True,
) -> np.ndarray:

    dataset = preprocess_dataframe(df)
    dataset = encode_features(dataset, feature_cols)

    X = dataset[feature_cols].values

    if scale:
        scaler = RobustScaler()
        X = scaler.fit_transform(X)

    return X


def save_model(model, path: str) -> None:
    joblib.dump(model, path)