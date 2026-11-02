
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# Page settings
st.set_page_config(
    page_title="Starbucks Calorie Predictor",
    page_icon="☕",
    layout="centered"
)

# Load trained model
MODEL_PATH = Path(__file__).parent / "starbucks_food_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.title("☕ Starbucks Food Calorie Predictor")
st.write("Enter the nutrition values to predict the calorie category.")

try:
    model = load_model()

    # These are the four features used in your food model
    default_features = [
        "Fat (g)",
        "Carb. (g)",
        "Fiber (g)",
        "Protein (g)"
    ]

    # Use the exact feature names stored in the trained model, if available
    if hasattr(model, "feature_names_in_"):
        feature_names = list(model.feature_names_in_)
    else:
        feature_names = default_features

    st.subheader("Enter Nutrition Information")

    values = {}

    for feature in feature_names:
        values[feature] = st.number_input(
            feature,
            min_value=0.0,
            value=0.0,
            step=1.0
        )

    if st.button("Predict Calorie Category"):
        input_data = pd.DataFrame([values], columns=feature_names)

        prediction = model.predict(input_data)[0]

        st.subheader("Prediction Result")

        if str(prediction).lower() == "high calorie":
            st.error("High Calorie")
        elif str(prediction).lower() == "low calorie":
            st.success("Low Calorie")
        else:
            st.info(f"Predicted Category: {prediction}")

        # Show prediction confidence when supported by the model
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            st.write("Prediction Confidence:")
            for label, probability in zip(model.classes_, probabilities):
                st.write(f"{label}: {probability * 100:.2f}%")

except FileNotFoundError:
    st.error(
        "Model file not found. Make sure "
        "starbucks_food_model.joblib is in the repository."
    )

except Exception as e:
    st.error(f"An error occurred: {e}")
