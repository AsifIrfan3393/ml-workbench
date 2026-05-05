from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif


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


# 🔥 Better preprocessing
def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for col in cleaned.columns:
        if cleaned[col].dtype == "object":
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])
        else:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    return cleaned


# 🔥 Encoding categorical features
def encode_features(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    encoded = df.copy()

    for col in columns:
        if encoded[col].dtype == "object":
            le = LabelEncoder()
            encoded[col] = le.fit_transform(encoded[col].astype(str))

    return encoded


# 🚀 MAIN IMPROVED FUNCTION
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

    # 🔥 Stratified split (important for classification)
    stratify = y if len(np.unique(y)) < 20 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=stratify
    )

    # 🔥 Feature selection (boost accuracy)
    if len(feature_cols) > 3:
        selector = SelectKBest(score_func=f_classif, k=min(5, X_train.shape[1]))
        X_train = selector.fit_transform(X_train, y_train)
        X_test = selector.transform(X_test)

    # 🔥 Better scaling (handles outliers better)
    if scale:
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


# 🚀 Clustering improvement
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