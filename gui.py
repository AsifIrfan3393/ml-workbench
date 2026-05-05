from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, fetch_california_housing

from models import ModelFactory
from utils import (
    get_example_datasets,
    load_dataset,
    prepare_clustering_data,
    prepare_supervised_data,
    save_model,
)


class MLWorkbenchGUI:
    def __init__(self) -> None:
        st.set_page_config(page_title="ML Workbench", layout="wide")
        self.factory = ModelFactory()

    def _create_example_data(self) -> None:
        examples = get_example_datasets()
        if not examples["Iris Classification"].exists():
            iris = load_iris(as_frame=True)
            iris_df = iris.frame
            iris_df.columns = [c.replace(" (cm)", "").replace(" ", "_") for c in iris_df.columns]
            iris_df.to_csv(examples["Iris Classification"], index=False)

        if not examples["Housing Regression"].exists():
            housing = fetch_california_housing(as_frame=True)
            h_df = housing.frame
            h_df.to_csv(examples["Housing Regression"], index=False)

    def _sidebar(self) -> Dict[str, Any]:
        st.sidebar.title("⚙️ ML Controls")
        algorithm = st.sidebar.selectbox("Algorithm", self.factory.SUPPORTED)
        test_size = st.sidebar.slider("Train-Test Split (test %)", 10, 50, 20, 5) / 100
        scale = st.sidebar.checkbox("Feature scaling", value=True)

        params: Dict[str, Any] = {}
        if algorithm == "Logistic Regression":
            params["C"] = st.sidebar.slider("C", 0.01, 10.0, 1.0)
            params["max_iter"] = st.sidebar.slider("Max Iterations", 100, 1000, 300, 50)
            params["solver"] = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear"])
        elif algorithm == "Decision Tree":
            params["max_depth"] = st.sidebar.slider("Max Depth", 1, 30, 5)
            params["min_samples_split"] = st.sidebar.slider("Min Samples Split", 2, 20, 2)
        elif algorithm == "Random Forest":
            params["n_estimators"] = st.sidebar.slider("N Estimators", 10, 500, 100, 10)
            params["max_depth"] = st.sidebar.slider("Max Depth", 1, 30, 8)
            params["min_samples_split"] = st.sidebar.slider("Min Samples Split", 2, 20, 2)
        elif algorithm == "Support Vector Machine (SVM)":
            params["C"] = st.sidebar.slider("C", 0.01, 10.0, 1.0)
            params["kernel"] = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
            params["gamma"] = st.sidebar.selectbox("Gamma", ["scale", "auto"])
        elif algorithm == "K-Means Clustering":
            params["n_clusters"] = st.sidebar.slider("Number of Clusters", 2, 10, 3)

        return {"algorithm": algorithm, "test_size": test_size, "scale": scale, "params": params}

    def _plot_confusion_matrix(self, matrix, title: str):
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

    def _plot_regression(self, y_true, y_pred):
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_true, y_pred, alpha=0.7)
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], color="red", linestyle="--")
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Regression Predictions")
        st.pyplot(fig)

    def _plot_clusters(self, X, clusters):
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(X)
        fig, ax = plt.subplots(figsize=(6, 4))
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=clusters, cmap="tab10", alpha=0.8)
        ax.set_title("K-Means Clusters (PCA 2D)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        fig.colorbar(scatter)
        st.pyplot(fig)

    def run(self) -> None:
        self._create_example_data()

        st.title("🧠 Mini ML Workbench")
        st.markdown("### 🚀 Interactive Machine Learning Dashboard")
        st.caption("Train, tune, compare, and visualize popular ML models in one place.")

        controls = self._sidebar()

        col1, col2 = st.columns([2, 1])
        with col1:
            file = st.file_uploader("Upload CSV Dataset", type=["csv"])
        with col2:
            examples = get_example_datasets()
            example_name = st.selectbox("Or load an example", ["None"] + list(examples.keys()))

        dataset = None
        if example_name != "None":
            dataset = pd.read_csv(examples[example_name])
        elif file is not None:
            try:
                dataset = load_dataset(file)
            except Exception as exc:
                st.error(f"Could not load dataset: {exc}")

        if dataset is None:
            st.info("Upload a CSV or choose an example dataset to begin.")
            return

        st.subheader("Dataset Preview")
        st.dataframe(dataset.head(20), use_container_width=True)
        st.write(f"Shape: {dataset.shape[0]} rows × {dataset.shape[1]} columns")

        # 🔥 NEW: dataset info
        st.write("Column Types:")
        st.write(dataset.dtypes)

        all_cols: List[str] = list(dataset.columns)
        algorithm = controls["algorithm"]

        if algorithm == "K-Means Clustering":
            feature_cols = st.multiselect("Select Features", all_cols, default=all_cols[: min(4, len(all_cols))])
            target_col = None
        else:
            target_col = st.selectbox("Select Target Column", all_cols, index=len(all_cols) - 1)
            default_features = [c for c in all_cols if c != target_col][: min(5, len(all_cols)-1)]
            feature_cols = st.multiselect("Select Features", [c for c in all_cols if c != target_col], default=default_features)

        if not feature_cols:
            st.warning("Please select at least one feature.")
            return

        # 🔥 NEW: feature count
        st.write(f"🔍 Selected Features: {len(feature_cols)}")

        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model... please wait"):
                try:
                    if algorithm == "K-Means Clustering":
                        X = prepare_clustering_data(dataset, feature_cols, scale=controls["scale"])
                        result, clusters = self.factory.train_kmeans(controls["params"], X)
                        st.success("Training complete!")
                        st.metric("Training Time (sec)", f"{result.training_time_sec:.4f}")
                        st.metric("Inertia", f"{result.metrics['Inertia']:.4f}")
                        self._plot_clusters(X, clusters)
                    else:
                        X_train, X_test, y_train, y_test = prepare_supervised_data(
                            dataset,
                            feature_cols,
                            target_col,
                            controls["test_size"],
                            scale=controls["scale"],
                        )

                        result = self.factory.train_supervised(
                            algorithm,
                            controls["params"],
                            X_train,
                            y_train,
                            X_test,
                            y_test,
                        )

                        st.success("Training complete!")
                        st.metric("Training Time (sec)", f"{result.training_time_sec:.4f}")

                        metric_cols = st.columns(len(result.metrics))
                        for i, (k, v) in enumerate(result.metrics.items()):
                            metric_cols[i].metric(k, f"{v:.4f}")

                        if algorithm == "Linear Regression":
                            self._plot_regression(result.y_test, result.y_pred)
                        else:
                            self._plot_confusion_matrix(result.confusion, f"{algorithm} Confusion Matrix")

                    st.session_state["last_model"] = result.model
                    st.session_state["last_algo"] = algorithm
                    st.session_state["last_result"] = result

                except Exception as exc:
                    st.error("❌ Training failed. Check dataset and parameters.")
                    st.exception(exc)

        st.subheader("Model Comparison (current session)")
        history = st.session_state.get("history", [])

        if st.button("Add latest run to comparison"):
            if "last_model" in st.session_state:
                row = {"algorithm": st.session_state.get("last_algo", "N/A")}
                last_result = st.session_state.get("last_result")
                row.update({"time_sec": getattr(last_result, "training_time_sec", None)})
                if last_result is not None:
                    row.update(last_result.metrics)
                history.append(row)
                st.session_state["history"] = history

        if history:
            df_hist = pd.DataFrame(history)
            st.dataframe(df_hist, use_container_width=True)

            if "Accuracy" in df_hist.columns:
                best = df_hist.loc[df_hist["Accuracy"].idxmax()]
                st.success(f"🏆 Best Model: {best['algorithm']} (Accuracy: {best['Accuracy']:.4f})")
                st.bar_chart(df_hist.set_index("algorithm")["Accuracy"])

            if "time_sec" in df_hist.columns:
                st.bar_chart(df_hist.set_index("algorithm")["time_sec"])

        st.subheader("Save Trained Model")
        save_name = st.text_input("Model filename", value="trained_model.joblib")

        if st.button("Save Model"):
            model = st.session_state.get("last_model")
            if model is None:
                st.warning("No trained model found. Train first.")
            else:
                path = Path(save_name)
                save_model(model, str(path))
                st.success(f"Model saved to: {path.resolve()}")