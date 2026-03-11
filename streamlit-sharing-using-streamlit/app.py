import streamlit as st
import joblib
import numpy as np
from pathlib import Path


@st.cache_resource
def load_model():
    app_directory = Path(__file__).resolve().parent
    project_root = app_directory.parent

    candidate_paths = [
        app_directory / "model" / "decisiontree_classifier_baseline.pkl",
        project_root / "model" / "decisiontree_classifier_baseline.pkl"
    ]

    for model_path in candidate_paths:
        if model_path.exists():
            return joblib.load(model_path), model_path

    raise FileNotFoundError(
        "Model file not found. Expected decisiontree_classifier_baseline.pkl in ./model or ../model"
    )

# Streamlit page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn using a Decision Tree Classifier.")

try:
    model, loaded_model_path = load_model()
    st.caption(f"Loaded model: {loaded_model_path}")
except Exception as exc:
    st.error(f"Failed to load model: {exc}")
    st.stop()

# Input form
with st.form("prediction_form"):
    monthly_fee = st.number_input("Monthly Fee", min_value=0.0, step=1.0)
    customer_age = st.number_input("Customer Age", min_value=0, step=1)
    support_calls = st.number_input("Support Calls", min_value=0, step=1)

    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        X = np.array([[monthly_fee, customer_age, support_calls]])
        prediction = model.predict(X)
        st.success(f"### Predicted Class: {int(prediction[0])}")
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
