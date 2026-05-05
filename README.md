# Mini ML Studio (Streamlit)

Mini ML Studio is a beginner-friendly but professional end-to-end GUI application to run and visualize multiple machine learning algorithms.

## Features

- Clean Streamlit GUI with sidebar controls
- CSV upload + built-in example dataset
- Dataset preview and feature/target selection
- Algorithms:
  - Linear Regression
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - K-Means Clustering
- Hyperparameter controls (sliders, dropdowns, checkboxes)
- Train-test split control
- Missing-value handling
- Metrics and visualization panel
- Confusion matrix for classification
- Regression actual-vs-predicted plot
- Cluster scatter plot
- Training-time tracking
- Save trained model to disk

## Project Structure

```text
Jscipt/
├── main.py
├── gui.py
├── models.py
├── utils.py
├── requirements.txt
├── README.md
└── data/
    └── sample_iris.csv
```

## Algorithm Explanations

### 1) Linear Regression
- **Type:** Supervised regression
- **Goal:** Predict a continuous value by learning a linear relationship between input features and target.
- **Metric used:** MSE, R2

### 2) Logistic Regression
- **Type:** Supervised classification
- **Goal:** Predict class probability and class label with a linear decision boundary (in transformed space).
- **Metrics used:** Accuracy, Precision, Recall, F1

### 3) Decision Tree
- **Type:** Supervised classification
- **Goal:** Split data into decision rules (if/else style) to classify samples.
- **Metrics used:** Accuracy, Precision, Recall, F1

### 4) Random Forest
- **Type:** Ensemble classification
- **Goal:** Combine many decision trees to improve robustness and reduce overfitting.
- **Metrics used:** Accuracy, Precision, Recall, F1

### 5) Support Vector Machine (SVM)
- **Type:** Supervised classification
- **Goal:** Find the maximum-margin hyperplane separating classes (kernel trick for non-linear separation).
- **Metrics used:** Accuracy, Precision, Recall, F1

### 6) K-Means Clustering
- **Type:** Unsupervised clustering
- **Goal:** Group data into K clusters by minimizing distance to cluster centroids.
- **Metric shown:** Inertia

## Installation

1. **Create and activate a virtual environment** (recommended):

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\\Scripts\\activate  # Windows
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

## How to Run

```bash
streamlit run main.py
```

Then open the local URL shown in your terminal.

## Step-by-step Usage

1. Launch app: `streamlit run main.py`
2. In sidebar, choose an algorithm.
3. Upload your CSV file or select the built-in Iris sample.
4. Select feature columns and target column.
5. Adjust hyperparameters.
6. Choose train-test split.
7. Click **Train Model**.
8. Review metrics, plots, and training time.
9. Optionally save the trained model to a `.joblib` file.

## Notes for Dataset Format

- Classification/regression models require a valid target column.
- K-Means ignores target and clusters by selected features.
- Numeric and categorical features are both supported.
- Missing values are handled automatically (median/mode imputation in-app).

## Error Handling Included

- Invalid CSV load errors
- Empty feature selection errors
- Automatic removal of target from features (with warning)
- Missing value handling
- Training exceptions shown in GUI

## Future Improvements (Optional)

- Add model comparison history table with multi-run tracking
- Add ROC/AUC and feature importance visuals
- Add downloadable predictions
- Add cross-validation controls
