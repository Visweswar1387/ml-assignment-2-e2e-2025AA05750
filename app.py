import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay
)
# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="ML Assignment 2",
    layout="centered"
)
st.title("ML Assignment 2")
st.write(
    """
    Upload **test data only (CSV format)**, select a trained model, and view evaluation metrics and confusion matrix.
    """
)
# --------------------------------------------------
# Load models
# --------------------------------------------------
MODEL_PATHS = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "KNN": "model/knn.pkl",
    # "Naive Bayes": "saved_models/naive_bayes.pkl",
    # "Random Forest (Ensemble)": "saved_models/random_forest.pkl",
    # "XGBoost (Ensemble)": "saved_models/xgboost.pkl"
}
EXPECTED_COLUMNS = [
    "age",
    "sex",
    "chest_pain_type",
    "resting_blood_pressure",
    "cholestoral",
    "fasting_blood_sugar",
    "rest_ecg",
    "Max_heart_rate",
    "exercise_induced_angina",
    "oldpeak",
    "slope",
    "vessels_colored_by_flourosopy",
    "thalassemia",
    "target"
]
@st.cache_resource
def load_model(path):
    return joblib.load(path)
# --------------------------------------------------
# Dataset upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload test dataset (CSV only)",
    type=["csv"]
)

def validate_dataSet(data):
    uploaded_cols = data.columns.tolist()
    missing = set(EXPECTED_COLUMNS) - set(uploaded_cols)
    extra = set(uploaded_cols) - set(EXPECTED_COLUMNS)

    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    if extra:
        st.warning(f"Extra columns will be ignored: {extra}")


if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Test Data")
    st.dataframe(data.head())
    validate_dataSet(data)

    X_test = data.drop("target", axis=1)
    y_test = data["target"]

    # --------------------------------------------------
    # Model selection
    # --------------------------------------------------
    model_name = st.selectbox(
        "Select Machine Learning Model",
        list(MODEL_PATHS.keys())
    )

    model = load_model(MODEL_PATHS[model_name])

    # --------------------------------------------------
    # Predictions
    # --------------------------------------------------
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # --------------------------------------------------
    # Evaluation metrics
    # --------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{accuracy:.4f}")
    col1.metric("AUC", f"{auc:.4f}")

    col2.metric("Precision", f"{precision:.4f}")
    col2.metric("Recall", f"{recall:.4f}")

    col3.metric("F1 Score", f"{f1:.4f}")
    col3.metric("MCC", f"{mcc:.4f}")

    # --------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap="Blues")
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file containing test data.")