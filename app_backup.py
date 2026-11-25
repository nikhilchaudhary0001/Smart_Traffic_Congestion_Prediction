
import json
from datetime import date, datetime, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Smart Traffic Congestion Prediction",
    page_icon="🚦",
    layout="wide"
)

st.title("Smart Traffic Congestion Prediction")
st.subheader("Random Forest Regression Prototype")

st.write(
    "Predict the next-hour probe count for a road segment "
    "using historical traffic observations and road characteristics."
)


# ==================================================
# 2. LOAD SAVED MODEL ARTIFACTS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "saved_model"

MODEL_PATH = MODEL_DIR / "random_forest_model.joblib"
FEATURES_PATH = MODEL_DIR / "model_features.json"
METRICS_PATH = MODEL_DIR / "model_metrics.json"


@st.cache_resource
def load_model_artifacts():
    required_paths = [
        MODEL_PATH,
        FEATURES_PATH,
        METRICS_PATH
    ]

    missing = [
        path.name for path in required_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required files are missing from saved_model: "
            + ", ".join(missing)
        )

    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        feature_names = json.load(file)

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        metrics = json.load(file)

    if len(feature_names) != model.n_features_in_:
        raise ValueError(
            "Saved feature count does not match the model."
        )

    return model, feature_names, metrics


try:
    model, feature_names, metrics = load_model_artifacts()
    st.success("Saved Random Forest model loaded successfully.")

except Exception as error:
    st.error(f"Unable to load model artifacts: {error}")
    st.stop()


# ==================================================
# 3. DISPLAY MODEL PERFORMANCE
# ==================================================

st.markdown("### Saved Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric("MAE", f"{metrics['MAE']:.4f}")
col2.metric("RMSE", f"{metrics['RMSE']:.4f}")
col3.metric("R²", f"{metrics['R2']:.4f}")

st.caption(
    "These metrics are loaded from the saved evaluation results. "
    "The model is not retrained by this application."
)


# ==================================================
# 4. COLLECT PREDICTION INPUTS
# ==================================================

st.markdown("### Enter Traffic Information")

with st.form("prediction_form"):

    st.markdown("#### Observation date and time")

    date_col, time_col = st.columns(2)

    with date_col:
        observation_date = st.date_input(
            "Current observation date",
            value=date.today()
        )

    with time_col:
        observation_time = st.time_input(
            "Current observation hour",
            value=time(12, 0)
        )

    st.markdown("#### Probe-count history")

    left, right = st.columns(2)

    with left:
        current_count = st.number_input(
            "Current-hour probe count",
            min_value=0.0,
            value=100.0,
            step=1.0
        )

        lag_1 = st.number_input(
            "Probe count 1 hour earlier",
            min_value=0.0,
            value=95.0,
            step=1.0
        )

        lag_2 = st.number_input(
            "Probe count 2 hours earlier",
            min_value=0.0,
            value=90.0,
            step=1.0
        )

    with right:
        lag_3 = st.number_input(
            "Probe count 3 hours earlier",
            min_value=0.0,
            value=85.0,
            step=1.0
        )

        lag_24 = st.number_input(
            "Probe count 24 hours earlier",
            min_value=0.0,
            value=80.0,
            step=1.0
        )

    st.markdown("#### Road-segment characteristics")

    road1, road2, road3 = st.columns(3)

    with road1:
        speed_limit = st.number_input(
            "Speed limit (dataset units)",
            min_value=0.0,
            value=50.0,
            step=1.0
        )

    with road2:
        frc = st.number_input(
            "Functional road class (frc)",
            min_value=0.0,
            value=3.0,
            step=1.0
        )

    with road3:
        distance = st.number_input(
            "Segment distance (dataset units)",
            min_value=0.0,
            value=1.0,
            step=0.1
        )

    submitted = st.form_submit_button(
        "Predict Next-Hour Probe Count",
        use_container_width=True
    )


# ==================================================
# 5. FEATURE ENGINEERING
# ==================================================

if submitted:

    timestamp = datetime.combine(
        observation_date,
        observation_time
    )

    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    is_weekend = int(day_of_week >= 5)

    # Cyclical encoding of hour and day of week.
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    day_of_week_sin = np.sin(
        2 * np.pi * day_of_week / 7
    )

    day_of_week_cos = np.cos(
        2 * np.pi * day_of_week / 7
    )

    # Match the notebook's shifted three-observation window.
    # The current observation is excluded.
    previous_counts = np.array(
        [lag_1, lag_2, lag_3],
        dtype=float
    )

    rolling_mean_3h = float(
        np.mean(previous_counts)
    )

    # pandas rolling.std() uses ddof=1 by default.
    rolling_std_3h = float(
        np.std(previous_counts, ddof=1)
    )

    # Create the features used by the trained model.
    feature_values = {
        "probeCount": current_count,
        "probeCount_lag_1": lag_1,
        "probeCount_lag_2": lag_2,
        "probeCount_lag_3": lag_3,
        "probeCount_lag_24": lag_24,
        "probeCount_rolling_mean_3h": rolling_mean_3h,
        "probeCount_rolling_std_3h": rolling_std_3h,
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "day_of_week_sin": day_of_week_sin,
        "day_of_week_cos": day_of_week_cos,
        "speedLimit": speed_limit,
        "frc": frc,
        "distance": distance
    }

    # Preserve the exact feature order from model metadata.
    try:
        input_df = pd.DataFrame(
            [[feature_values[name] for name in feature_names]],
            columns=feature_names
        )
    except KeyError as error:
        st.error(f"Required model feature is missing: {error}")
        st.stop()

    if not np.isfinite(input_df.to_numpy(dtype=float)).all():
        st.error("The input features contain invalid values.")
        st.stop()


    # ==================================================
    # 6. GENERATE PREDICTION
    # ==================================================

    try:
        prediction = float(model.predict(input_df)[0])

        st.markdown("---")
        st.markdown("### Prediction Result")

        result1, result2 = st.columns(2)

        result1.metric(
            "Predicted next-hour probe count",
            f"{prediction:,.2f}"
        )

        result2.write("**Observation timestamp**")
        result2.write(timestamp.strftime("%Y-%m-%d %H:%M"))
        result2.write("**Forecast horizon**")
        result2.write("Next hourly observation")

        st.info(
            "This model predicts a numerical probe count. "
            "It does not directly predict a formal congestion category "
            "or an exact number of vehicles."
        )

        with st.expander("View all 17 model input features"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction failed: {error}")


# ==================================================
# 7. PROJECT INFORMATION
# ==================================================

with st.expander("About this project"):

    st.write(
        "**Project:** Smart Traffic Congestion Prediction "
        "Using Machine Learning"
    )

    st.write(
        "**Dataset:** New Delhi Traffic Probe & Analytics Dataset (2024)"
    )

    st.write("**Selected model:** Random Forest Regression")

    st.write(
        "**Target:** Next-hour probe count for the same road segment"
    )
