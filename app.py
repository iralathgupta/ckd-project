# pip install streamlit pandas requests json

import streamlit as st
import pandas as pd
import requests
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from ucimlrepo import fetch_ucirepo
import numpy as np

API_URL = "https://ckd-project-0267.onrender.com/predict"

FEATURE_ORDER = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
    'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc',
    'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="CKD Risk Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 CKD Risk Prediction System")
st.caption("Cloud-deployed explainable ML system for early detection of Chronic Kidney Disease")
st.divider()

# ── Load test data (cached so it only runs once) ─────────────
@st.cache_data
def load_test_data():
    ckd = fetch_ucirepo(id=336)
    df = pd.concat([ckd.data.features, ckd.data.targets], axis=1)
    df['class'] = df['class'].str.strip()

    cat_cols = df.select_dtypes(include='object').columns.tolist()
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()

    df_encoded = df.copy()
    le = LabelEncoder()
    for col in cat_cols:
        df_encoded[col] = df_encoded[col].fillna('missing')
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    imputer = KNNImputer(n_neighbors=5)
    df_imputed_array = imputer.fit_transform(df_encoded)
    df_imputed = pd.DataFrame(df_imputed_array, columns=df.columns)

    X = df_imputed.drop('class', axis=1)
    y = df_imputed['class']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_test.reset_index(drop=True), y_test.reset_index(drop=True)

X_test, y_test = load_test_data()

# ── Sidebar controls ─────────────────────────────────────────
with st.sidebar:
    st.header("Settings")

    # Toggle 1: View mode
    st.subheader("View Mode")
    view_mode = st.radio(
        "Who is viewing this result?",
        options=["patient", "clinic"],
        format_func=lambda x: "🧑 Patient View" if x == "patient" else "🏥 Clinic View",
        help="Patient view uses plain English. Clinic view includes ICD-10 codes, PubMed citations, and clinical trial matches via MCP."
    )

    st.divider()

    # Toggle 2: Input mode
    st.subheader("Input Mode")
    input_mode = st.radio(
        "How would you like to provide patient data?",
        options=["test", "manual"],
        format_func=lambda x: "📋 Test Dataset Record" if x == "test" else "✏️ Manual JSON Input"
    )

# ── Input section ─────────────────────────────────────────────
st.subheader("Patient Data")

patient_data = None

if input_mode == "test":
    col1, col2 = st.columns([2, 1])

    with col1:
        patient_index = st.slider(
            "Select a patient from the test dataset",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0,
            help=f"80 patients in the test set (indices 0–{len(X_test)-1})"
        )

    with col2:
        actual_label = "CKD" if y_test.iloc[patient_index] == 1 else "No CKD"
        st.metric(
            label="Actual diagnosis (ground truth)",
            value=actual_label,
            help="This is the confirmed diagnosis from the dataset — used to check model accuracy"
        )

    # Show the selected patient's data as a readable table
    selected = X_test.iloc[patient_index]
    st.dataframe(
        selected.to_frame(name="Value").T,
        use_container_width=True
    )

    patient_data = selected.to_dict()

elif input_mode == "manual":
    st.caption("Enter patient biomarkers as a JSON object. Only the 5 mandatory fields are required — leave others as null.")

    # Build a template JSON with mandatory fields filled and optional as null
    template = {
        "hemo": 11.2,
        "sg": 1.015,
        "sc": 1.2,
        "al": 1.0,
        "pcv": 38.0,
        "age": None,
        "bp": None,
        "bgr": None,
        "bu": None,
        "sod": None,
        "htn": None,
        "dm": None,
        "su": None,
        "rbc": None,
        "pc": None,
        "pcc": None,
        "ba": None,
        "pot": None,
        "wbcc": None,
        "rbcc": None,
        "cad": None,
        "appet": None,
        "pe": None,
        "ane": None
    }

    json_input = st.text_area(
        label="Patient data (JSON)",
        value=json.dumps(template, indent=2),
        height=420,
        help="Mandatory: hemo, sg, sc, al, pcv. All others optional."
    )

    # Validate JSON in real time
    try:
        patient_data = json.loads(json_input)
        st.success("✓ Valid JSON")
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON: {e}")
        patient_data = None

# ── Run prediction ────────────────────────────────────────────
st.divider()

run = st.button(
    "Run Prediction",
    type="primary",
    disabled=patient_data is None,
    use_container_width=True
)

if run and patient_data is not None:
    with st.spinner("Running model and generating explanation..."):
        try:
            response = requests.post(
                API_URL,
                json=patient_data,
                params={"view": view_mode},
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()

                st.divider()

                # ── Result header
                pred = result["prediction"]
                conf = result["confidence"]
                ckd_prob = result["ckd_probability"]

                col1, col2, col3 = st.columns(3)
                col1.metric("Prediction", pred)
                col2.metric("Confidence", f"{conf:.1%}")
                col3.metric("CKD Probability", f"{ckd_prob:.1%}")

                st.divider()

                # ── SHAP contributions chart
                st.subheader("Top Contributing Features")
                shap_data = result["shap_contributions"]
                shap_df = pd.DataFrame(
                    list(shap_data.items()),
                    columns=["Feature", "SHAP Value"]
                ).sort_values("SHAP Value", key=abs, ascending=True)

                import plotly.express as px
                fig = px.bar(
                    shap_df,
                    x="SHAP Value",
                    y="Feature",
                    orientation="h",
                    color="SHAP Value",
                    color_continuous_scale=["#1D9E75", "#ffffff", "#D85A30"],
                    color_continuous_midpoint=0,
                    title="Feature contributions (red = toward CKD, green = away from CKD)"
                )
                fig.update_layout(
                    height=300,
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                st.plotly_chart(fig, use_container_width=True)

                st.divider()

                # ── Explanation
                mode_label = "🧑 Patient Explanation" if view_mode == "patient" else "🏥 Clinical Summary"
                st.subheader(mode_label)
                st.write(result["explanation"])

                # ── View toggle reminder
                other = "clinic" if view_mode == "patient" else "patient"
                other_label = "Clinic View" if other == "clinic" else "Patient View"
                st.info(f"Switch to **{other_label}** in the sidebar to see the {'detailed clinical summary with ICD-10 codes and PubMed citations' if other == 'clinic' else 'simplified patient-friendly explanation'}.")

            else:
                st.error(f"API error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to API. Make sure `uvicorn api:app --reload` is running.")
        except requests.exceptions.Timeout:
            st.error("Request timed out. The clinic view MCP lookup can take up to 30 seconds — try again.")