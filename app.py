import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="CKD Risk Predictor",
    page_icon="assets/favicon.png" if os.path.exists("assets/favicon.png") else None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Styling ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Hide default Streamlit chrome */
    #MainMenu { visibility: hidden; }
    footer     { visibility: hidden; }
    header     { visibility: hidden; }

    /* Page background */
    .stApp {
        background-color: #F7F5F0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1A1A18;
        border-right: 1px solid #2E2E2B;
    }
    section[data-testid="stSidebar"] * {
        color: #C8C5BC !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #C8C5BC !important;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #F7F5F0 !important;
        font-family: 'DM Serif Display', serif !important;
    }

    /* Page title */
    .page-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.4rem;
        color: #1A1A18;
        margin: 0 0 0.25rem;
        line-height: 1.1;
    }
    .page-subtitle {
        font-size: 0.95rem;
        color: #7A7870;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    /* Cards */
    .card {
        background: #FFFFFF;
        border: 1px solid #E5E2DA;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-family: 'DM Serif Display', serif;
        font-size: 1.1rem;
        color: #1A1A18;
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid #E5E2DA;
    }

    /* Result banner */
    .result-ckd {
        background: #FDF2EE;
        border: 1px solid #E8C4B4;
        border-left: 4px solid #C45C2E;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .result-healthy {
        background: #EFF7F3;
        border: 1px solid #B4D9C4;
        border-left: 4px solid #2E7D52;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1rem;
    }
    .result-label {
        font-family: 'DM Serif Display', serif;
        font-size: 1.4rem;
        margin: 0 0 0.25rem;
    }
    .result-conf {
        font-size: 0.85rem;
        color: #7A7870;
        margin: 0;
    }

    /* Metric boxes */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.25rem;
    }
    .metric-box {
        flex: 1;
        background: #F7F5F0;
        border: 1px solid #E5E2DA;
        border-radius: 8px;
        padding: 0.85rem 1rem;
        text-align: center;
    }
    .metric-value {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        color: #1A1A18;
        line-height: 1;
        margin-bottom: 0.2rem;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #7A7870;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* SHAP bars */
    .shap-row {
        display: flex;
        align-items: center;
        margin-bottom: 0.5rem;
        gap: 0.75rem;
    }
    .shap-feat {
        width: 80px;
        font-size: 0.8rem;
        color: #1A1A18;
        text-align: right;
        flex-shrink: 0;
        font-family: 'DM Sans', monospace;
    }
    .shap-bar-wrap {
        flex: 1;
        height: 20px;
        background: #F0EDE6;
        border-radius: 3px;
        position: relative;
        overflow: hidden;
    }
    .shap-bar-pos {
        position: absolute;
        left: 50%;
        top: 0;
        height: 100%;
        background: #C45C2E;
        border-radius: 0 3px 3px 0;
    }
    .shap-bar-neg {
        position: absolute;
        right: 50%;
        top: 0;
        height: 100%;
        background: #2E7D52;
        border-radius: 3px 0 0 3px;
    }
    .shap-val {
        width: 55px;
        font-size: 0.78rem;
        color: #7A7870;
        flex-shrink: 0;
        font-family: 'DM Sans', monospace;
    }

    /* Explanation text */
    .explanation-box {
        background: #F7F5F0;
        border: 1px solid #E5E2DA;
        border-radius: 8px;
        padding: 1.25rem;
        font-size: 0.95rem;
        line-height: 1.75;
        color: #2C2C2A;
        white-space: pre-wrap;
    }

    /* Divider */
    .section-divider {
        border: none;
        border-top: 1px solid #E5E2DA;
        margin: 1.5rem 0;
    }

    /* JSON input */
    .stTextArea textarea {
        font-family: 'DM Sans', monospace !important;
        font-size: 0.82rem !important;
        background: #FAFAF8 !important;
        border: 1px solid #E5E2DA !important;
        border-radius: 8px !important;
    }

    /* Buttons */
    .stButton button {
        background-color: #1A1A18 !important;
        color: #F7F5F0 !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.5rem !important;
        width: 100% !important;
        transition: background 0.2s !important;
    }
    .stButton button:hover {
        background-color: #2E2E2B !important;
    }

    /* Slider */
    .stSlider {
        padding-top: 0.5rem;
    }

    /* Warning / error */
    .stAlert {
        border-radius: 8px !important;
    }

    /* Ground truth badge */
    .gt-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 99px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    .gt-ckd     { background: #FDF2EE; color: #C45C2E; border: 1px solid #E8C4B4; }
    .gt-healthy { background: #EFF7F3; color: #2E7D52; border: 1px solid #B4D9C4; }

    /* Mode toggle label */
    .mode-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #7A7870;
        margin-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/predict")

FEATURE_ORDER = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
    'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc',
    'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]

MANDATORY_FIELDS = ['hemo', 'sg', 'sc', 'al', 'pcv']

FEATURE_LABELS = {
    'age': 'Age', 'bp': 'Blood Pressure', 'sg': 'Specific Gravity',
    'al': 'Albumin', 'su': 'Sugar', 'rbc': 'Red Blood Cells (urine)',
    'pc': 'Pus Cells', 'pcc': 'Pus Cell Clumps', 'ba': 'Bacteria',
    'bgr': 'Blood Glucose Random', 'bu': 'Blood Urea', 'sc': 'Serum Creatinine',
    'sod': 'Sodium', 'pot': 'Potassium', 'hemo': 'Haemoglobin',
    'pcv': 'Packed Cell Volume', 'wbcc': 'White Blood Cell Count',
    'rbcc': 'Red Blood Cell Count', 'htn': 'Hypertension',
    'dm': 'Diabetes Mellitus', 'cad': 'Coronary Artery Disease',
    'appet': 'Appetite', 'pe': 'Pedal Edema', 'ane': 'Anaemia'
}

# ── Load test data ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset...")
def load_test_data():
    try:
        from ucimlrepo import fetch_ucirepo
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.impute import KNNImputer

        ckd = fetch_ucirepo(id=336)
        df  = pd.concat([ckd.data.features, ckd.data.targets], axis=1)
        df['class'] = df['class'].str.strip()

        cat_cols = df.select_dtypes(include='object').columns.tolist()
        df_encoded = df.copy()
        le = LabelEncoder()
        for col in cat_cols:
            df_encoded[col] = df_encoded[col].fillna('missing')
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

        from sklearn.impute import KNNImputer
        imputer    = KNNImputer(n_neighbors=5)
        df_imputed = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df.columns)

        X = df_imputed.drop('class', axis=1)
        y = df_imputed['class']
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        return X_test.reset_index(drop=True), y_test.reset_index(drop=True), None

    except Exception as e:
        return None, None, str(e)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## CKD Predictor")
    st.markdown("---")

    st.markdown("### View Mode")
    st.markdown(
        '<p style="font-size:0.78rem;color:#888;margin-bottom:0.4rem">'
        'Patient view uses plain language. Clinic view includes ICD-10 codes, '
        'PubMed citations, and drug contraindications via MCP.'
        '</p>',
        unsafe_allow_html=True
    )
    view_mode = st.radio(
        label="view_mode",
        options=["patient", "clinic"],
        format_func=lambda x: "Patient View" if x == "patient" else "Clinic View",
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown("### Input Mode")
    input_mode = st.radio(
        label="input_mode",
        options=["test", "manual"],
        format_func=lambda x: "Test Dataset Record" if x == "test" else "Manual JSON Input",
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(
        '<p style="font-size:0.75rem;color:#666;line-height:1.6">'
        'Model: Random Forest<br>'
        'XAI: SHAP (global) + LIME (local)<br>'
        'NL Reasoning: Claude Sonnet<br>'
        'Dataset: UCI CKD (400 samples)<br>'
        'MCP: healthcare-mcp-public server by Cicatriiz'
        '</p>',
        unsafe_allow_html=True
    )

# ── Main content ──────────────────────────────────────────────────────────────

st.markdown('<h1 class="page-title">Kidney Disease Risk Predictor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="page-subtitle">'
    'Cloud-deployed explainable ML system for early detection of Chronic Kidney Disease'
    '</p>',
    unsafe_allow_html=True
)

# ── Input section ─────────────────────────────────────────────────────────────

patient_data = None
input_error  = None

if input_mode == "test":
    X_test, y_test, load_error = load_test_data()

    if load_error:
        st.error(f"Failed to load test dataset: {load_error}")
    elif X_test is not None:
        col_slider, col_info = st.columns([3, 1])

        with col_slider:
            patient_index = st.slider(
                label    = "Select patient from test dataset",
                min_value= 0,
                max_value= len(X_test) - 1,
                value    = 0,
                help     = f"{len(X_test)} patients available (indices 0 to {len(X_test) - 1})"
            )

        with col_info:
            actual_raw   = y_test.iloc[patient_index]
            actual_label = "CKD" if actual_raw == 1 else "No CKD"
            badge_class  = "gt-ckd" if actual_raw == 1 else "gt-healthy"
            st.markdown(
                f'<p style="font-size:0.75rem;color:#7A7870;margin-bottom:0.3rem">Ground truth</p>'
                f'<span class="gt-badge {badge_class}">{actual_label}</span>',
                unsafe_allow_html=True
            )

        selected     = X_test.iloc[patient_index]
        patient_data = selected.to_dict()

        # Show patient data in a clean table
        display_df = selected.to_frame(name="Value").T.rename(
            columns=FEATURE_LABELS
        )
        st.dataframe(
            display_df.style.format("{:.2f}"),
            use_container_width=True,
            height=68
        )

elif input_mode == "manual":
    st.markdown(
        '<p style="font-size:0.85rem;color:#7A7870;margin-bottom:0.5rem">'
        'Enter patient biomarkers as JSON. Mandatory fields: '
        '<code>hemo</code>, <code>sg</code>, <code>sc</code>, '
        '<code>al</code>, <code>pcv</code>. Set optional fields to <code>null</code>.'
        '</p>',
        unsafe_allow_html=True
    )

    template = {
        "hemo": 11.2, "sg": 1.015, "sc": 1.2, "al": 1.0, "pcv": 38.0,
        "age": None, "bp": None, "bgr": None, "bu": None, "sod": None,
        "htn": None, "dm": None, "su": None, "rbc": None, "pc": None,
        "pcc": None, "ba": None, "pot": None, "wbcc": None, "rbcc": None,
        "cad": None, "appet": None, "pe": None, "ane": None
    }

    json_input = st.text_area(
        label      = "Patient data (JSON)",
        value      = json.dumps(template, indent=2),
        height     = 400,
        label_visibility = "collapsed"
    )

    try:
        parsed       = json.loads(json_input)
        missing_mandatory = [f for f in MANDATORY_FIELDS if parsed.get(f) is None]

        if missing_mandatory:
            input_error = f"Missing mandatory fields: {', '.join(missing_mandatory)}"
            st.error(input_error)
        else:
            patient_data = parsed
            st.success("Valid JSON — all mandatory fields present")

    except json.JSONDecodeError as e:
        input_error = f"Invalid JSON: {e}"
        st.error(input_error)

# ── Run button ────────────────────────────────────────────────────────────────

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

run_disabled = (patient_data is None) or (input_error is not None)

if st.button("Run Prediction", disabled=run_disabled):
    with st.spinner("Running model and generating explanation..."):
        try:
            response = requests.post(
                API_URL,
                json    = patient_data,
                params  = {"view": view_mode},
                timeout = 90
            )

            if response.status_code == 200:
                result = response.json()

                pred        = result.get("prediction", "Unknown")
                conf        = result.get("confidence", 0)
                ckd_prob    = result.get("ckd_probability", 0)
                explanation = result.get("explanation", "No explanation returned.")
                shap_data   = result.get("shap_contributions", {})
                lime_data   = result.get("lime_contributions", {})
                is_ckd      = "CKD detected" in pred

                # ── Result banner
                banner_class = "result-ckd" if is_ckd else "result-healthy"
                banner_color = "#C45C2E"    if is_ckd else "#2E7D52"
                st.markdown(
                    f'<div class="{banner_class}">'
                    f'<p class="result-label" style="color:{banner_color}">{pred}</p>'
                    f'<p class="result-conf">Confidence: {conf:.1%} &nbsp;|&nbsp; '
                    f'CKD probability: {ckd_prob:.1%} &nbsp;|&nbsp; '
                    f'View: {view_mode.title()}</p>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                col_left, col_right = st.columns([1, 1])

                # ── Left: SHAP contributions
                with col_left:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">SHAP Feature Contributions</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<p style="font-size:0.75rem;color:#7A7870;margin-bottom:0.75rem">'
                        'Red = pushes toward CKD &nbsp; Green = pushes away'
                        '</p>',
                        unsafe_allow_html=True
                    )

                    if shap_data:
                        max_abs = max(abs(v) for v in shap_data.values()) or 1
                        sorted_shap = sorted(shap_data.items(), key=lambda x: abs(x[1]), reverse=True)

                        for feat, val in sorted_shap:
                            bar_pct  = min(abs(val) / max_abs * 48, 48)
                            bar_html = ""
                            if val > 0:
                                bar_html = f'<div class="shap-bar-pos" style="width:{bar_pct}%"></div>'
                            else:
                                bar_html = f'<div class="shap-bar-neg" style="width:{bar_pct}%"></div>'

                            sign = "+" if val > 0 else ""
                            st.markdown(
                                f'<div class="shap-row">'
                                f'  <div class="shap-feat">{feat}</div>'
                                f'  <div class="shap-bar-wrap">{bar_html}</div>'
                                f'  <div class="shap-val">{sign}{val:.3f}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown('<p style="font-size:0.85rem;color:#7A7870">No SHAP data returned.</p>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                # ── Right: LIME contributions
                with col_right:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown('<div class="card-title">LIME Local Explanation</div>', unsafe_allow_html=True)
                    st.markdown(
                        '<p style="font-size:0.75rem;color:#7A7870;margin-bottom:0.75rem">'
                        'Local explanation specific to this patient only'
                        '</p>',
                        unsafe_allow_html=True
                    )

                    if lime_data:
                        max_abs_lime = max(abs(v) for v in lime_data.values()) or 1
                        sorted_lime  = sorted(lime_data.items(), key=lambda x: abs(x[1]), reverse=True)

                        for condition, weight in sorted_lime:
                            bar_pct  = min(abs(weight) / max_abs_lime * 48, 48)
                            bar_html = ""
                            if weight > 0:
                                bar_html = f'<div class="shap-bar-pos" style="width:{bar_pct}%"></div>'
                            else:
                                bar_html = f'<div class="shap-bar-neg" style="width:{bar_pct}%"></div>'

                            sign = "+" if weight > 0 else ""
                            # Truncate long condition strings
                            short_cond = condition if len(condition) <= 22 else condition[:20] + ".."
                            st.markdown(
                                f'<div class="shap-row">'
                                f'  <div class="shap-feat" style="width:130px;font-size:0.72rem">{short_cond}</div>'
                                f'  <div class="shap-bar-wrap">{bar_html}</div>'
                                f'  <div class="shap-val">{sign}{weight:.3f}</div>'
                                f'</div>',
                                unsafe_allow_html=True
                            )
                    else:
                        st.markdown('<p style="font-size:0.85rem;color:#7A7870">No LIME data returned.</p>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                thinking = result.get("thinking", "")

                if thinking and view_mode == "clinic":
                    with st.expander("Claude's reasoning process", expanded=False):
                        st.markdown(
                            f'<div style="'
                            f'font-family: monospace; font-size: 0.8rem; '
                            f'line-height: 1.7; color: #4A4A48; '
                            f'background: #F7F5F0; padding: 1rem; '
                            f'border-radius: 8px; white-space: pre-wrap;">'
                            f'{thinking}'
                            f'</div>',
                            unsafe_allow_html=True
        )
                # ── Explanation
                st.markdown('<div class="card">', unsafe_allow_html=True)
                mode_label = "Patient Explanation" if view_mode == "patient" else "Clinical Summary"
                st.markdown(f'<div class="card-title">{mode_label}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="explanation-box">{explanation}</div>',
                    unsafe_allow_html=True
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # ── Switch view nudge
                other       = "clinic"   if view_mode == "patient" else "patient"
                other_label = "Clinic View" if other == "clinic"   else "Patient View"
                other_desc  = (
                    "detailed clinical summary with ICD-10 codes, PubMed citations, and drug contraindications"
                    if other == "clinic"
                    else "plain-language summary for the patient"
                )
                st.info(f"Switch to {other_label} in the sidebar to see the {other_desc}.")

            else:
                st.error(f"API returned error {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            st.error(
                "Could not connect to the API. "
                "Make sure `uvicorn api:app --reload` is running on port 8000."
            )
        except requests.exceptions.Timeout:
            st.error(
                "The request timed out. The clinic view MCP lookup can take up to 60 seconds "
                "on first use while the server wakes up — please try again."
            )
        except Exception as e:
            st.error(f"Unexpected error: {e}")