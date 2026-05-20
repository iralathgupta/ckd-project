import os
import joblib
import numpy as np
import pandas as pd
import shap

from enum import Enum
from pathlib import Path
from string import Template
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from anthropic import Anthropic
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from ucimlrepo import fetch_ucirepo

# ── Environment ───────────────────────────────────────────────────────────────

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CKD Risk Prediction API",
    description="Cloud-deployed explainable ML system for early detection of Chronic Kidney Disease",
    version="1.0.0"
)

# ── Constants ─────────────────────────────────────────────────────────────────

FEATURE_ORDER = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
    'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc',
    'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]

PROMPT_FILE = Path(__file__).resolve().parent / "prompt_templates.txt"


def load_prompt_templates():
    raw_text = PROMPT_FILE.read_text(encoding="utf-8")
    templates = {}
    current_name = None
    current_lines = []

    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if current_name:
                templates[current_name] = "\n".join(current_lines).strip()
            current_name = stripped.strip("[]").lower()
            current_lines = []
        elif current_name is not None:
            current_lines.append(line)

    if current_name:
        templates[current_name] = "\n".join(current_lines).strip()

    return templates


PROMPT_TEMPLATES = load_prompt_templates()


def render_prompt(template_name: str, **kwargs) -> str:
    template_text = PROMPT_TEMPLATES.get(template_name)
    if template_text is None:
        raise ValueError(f"Prompt template '{template_name}' not found in {PROMPT_FILE}")
    return Template(template_text).substitute(**kwargs)


# ── Load model and explainers ─────────────────────────────────────────────────

model    = joblib.load("ckd_model.pkl")
medians  = joblib.load("feature_medians.pkl")
explainer = shap.TreeExplainer(model)

# ── Rebuild LIME explainer from training data ─────────────────────────────────
# LIME needs the training data distribution to generate local explanations

def build_lime_explainer():
    ckd = fetch_ucirepo(id=336)
    df  = pd.concat([ckd.data.features, ckd.data.targets], axis=1)
    df['class'] = df['class'].str.strip()

    cat_cols = df.select_dtypes(include='object').columns.tolist()

    df_encoded = df.copy()
    le = LabelEncoder()
    for col in cat_cols:
        df_encoded[col] = df_encoded[col].fillna('missing')
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

    imputer       = KNNImputer(n_neighbors=5)
    df_imputed    = pd.DataFrame(imputer.fit_transform(df_encoded), columns=df.columns)
    X             = df_imputed.drop('class', axis=1)
    y             = df_imputed['class']
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return LimeTabularExplainer(
        training_data  = X_train.values,
        feature_names  = FEATURE_ORDER,
        class_names    = ['notckd', 'ckd'],
        mode           = 'classification'
    )

lime_explainer = build_lime_explainer()

# ── View mode enum ────────────────────────────────────────────────────────────

class ViewMode(str, Enum):
    patient = "patient"
    clinic  = "clinic"

# ── Input schema ──────────────────────────────────────────────────────────────

class PatientData(BaseModel):
    # Mandatory
    hemo: float = Field(..., description="Haemoglobin level (g/dL)",         example=11.2)
    sg:   float = Field(..., description="Specific gravity",                  example=1.015)
    sc:   float = Field(..., description="Serum creatinine (mg/dL)",          example=1.2)
    al:   float = Field(..., description="Albumin in urine (0–5 scale)",      example=1.0)
    pcv:  float = Field(..., description="Packed cell volume (%)",            example=38.0)

    # Strongly recommended
    age:  Optional[float] = Field(None, description="Age in years")
    bp:   Optional[float] = Field(None, description="Blood pressure (mm/Hg)")
    bgr:  Optional[float] = Field(None, description="Blood glucose random (mg/dL)")
    bu:   Optional[float] = Field(None, description="Blood urea (mg/dL)")
    sod:  Optional[float] = Field(None, description="Sodium (mEq/L)")
    htn:  Optional[int]   = Field(None, description="Hypertension (0=no, 1=yes)")
    dm:   Optional[int]   = Field(None, description="Diabetes mellitus (0=no, 1=yes)")

    # Optional
    su:    Optional[float] = Field(None, description="Sugar (0–5 scale)")
    rbc:   Optional[int]   = Field(None, description="Red blood cells in urine (0=normal, 1=abnormal)")
    pc:    Optional[int]   = Field(None, description="Pus cells (0=normal, 1=abnormal)")
    pcc:   Optional[int]   = Field(None, description="Pus cell clumps (0=notpresent, 1=present)")
    ba:    Optional[int]   = Field(None, description="Bacteria (0=notpresent, 1=present)")
    pot:   Optional[float] = Field(None, description="Potassium (mEq/L)")
    wbcc:  Optional[float] = Field(None, description="White blood cell count (cells/cumm)")
    rbcc:  Optional[float] = Field(None, description="Red blood cell count (millions/cmm)")
    cad:   Optional[int]   = Field(None, description="Coronary artery disease (0=no, 1=yes)")
    appet: Optional[int]   = Field(None, description="Appetite (0=good, 1=poor)")
    pe:    Optional[int]   = Field(None, description="Pedal edema (0=no, 1=yes)")
    ane:   Optional[int]   = Field(None, description="Anaemia (0=no, 1=yes)")

# ── Helpers ───────────────────────────────────────────────────────────────────

def impute_missing(data: dict) -> dict:
    """Fill missing optional fields with training set medians."""
    for feat in FEATURE_ORDER:
        if data.get(feat) is None:
            data[feat] = medians[feat]
    return data


def extract_shap(df_input: pd.DataFrame):
    """Return a flat (n_features,) numpy array of SHAP values for one sample."""
    raw = explainer.shap_values(df_input)

    if isinstance(raw, list):
        # Random Forest — returns [class_0, class_1], each shape (1, n_features)
        vals = np.array(raw[1]).flatten()
    else:
        # XGBoost — returns shape (1, n_features)
        vals = np.array(raw).flatten()

    return vals[:len(FEATURE_ORDER)]


def extract_lime(df_input: pd.DataFrame, top_n: int = 5) -> list:
    """Return LIME feature-weight pairs for one sample."""
    exp = lime_explainer.explain_instance(
        df_input.iloc[0].values,
        model.predict_proba,
        num_features=top_n
    )
    return exp.as_list()


def build_shap_global_summary(shap_values_all, top_n: int = 10) -> str:
    """Mean absolute SHAP per feature across the whole test set."""
    mean_abs = np.abs(shap_values_all).mean(axis=0)
    ranked   = sorted(zip(FEATURE_ORDER, mean_abs), key=lambda x: x[1], reverse=True)[:top_n]

    lines = ["Global feature importance (averaged across all test patients):"]
    for i, (feat, imp) in enumerate(ranked, 1):
        lines.append(f"  {i}. {feat}: mean |SHAP| = {imp:.4f}")

    lines.append("\nDirection of top 3 features:")
    for feat, _ in ranked[:3]:
        idx  = FEATURE_ORDER.index(feat)
        mean = shap_values_all[:, idx].mean()
        direction = "generally increases CKD risk" if mean > 0 else "generally decreases CKD risk"
        lines.append(f"  - {feat}: {direction} (mean SHAP = {mean:.4f})")

    return "\n".join(lines)


def build_lime_local_summary(lime_pairs: list, outcome: str, confidence: float) -> str:
    lines = [
        f"Local explanation:",
        f"  Prediction : {outcome} (confidence: {confidence:.1%})",
        f"  Feature conditions that influenced this prediction:"
    ]
    for condition, weight in lime_pairs:
        direction = "toward CKD" if weight > 0 else "away from CKD"
        lines.append(f"  - {condition}: weight = {weight:.4f} ({direction})")
    return "\n".join(lines)

# ── Prompt builders ───────────────────────────────────────────────────────────

def build_patient_prompt(outcome: str, confidence: float, ckd_probability: float,
                         pairs: list, shap_lines: list) -> str:

    top_features = [f for f, s, v in pairs[:3]]
    plain_outcome = (
        "your kidneys may be showing signs of chronic kidney disease"
        if "CKD detected" in outcome
        else "your kidneys appear to be healthy"
    )

    return render_prompt(
        "patient",
        plain_outcome=plain_outcome,
        confidence=f"{confidence:.1%}",
        top_features=", ".join(top_features),
        shap_lines="\n".join(shap_lines)
    )


def build_clinic_prompt(outcome, confidence, ckd_probability,
                        pairs, shap_lines, lime_local, shap_global):
    patient_values = "\n".join([
        f"- {f}: {float(v):.2f} (SHAP contribution: {float(s):+.3f})"
        for f, s, v in pairs
    ])

    return render_prompt(
        "clinic",
        outcome=outcome,
        confidence=f"{confidence:.1%}",
        ckd_probability=f"{ckd_probability:.1%}",
        patient_values=patient_values,
        lime_local=lime_local,
        shap_global=shap_global
    )

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    key_loaded = os.getenv("ANTHROPIC_API_KEY") is not None
    return {"status": "ok", "api_key_loaded": key_loaded}


@app.post("/predict")
def predict(patient: PatientData, view: ViewMode = ViewMode.patient):

    # 1 — Impute and build dataframe
    data     = patient.dict()
    data     = impute_missing(data)
    df_input = pd.DataFrame([data])[FEATURE_ORDER]

    # 2 — Predict
    prediction      = int(model.predict(df_input)[0])
    proba           = model.predict_proba(df_input)[0]
    ckd_probability = float(proba[1])
    confidence      = ckd_probability if prediction == 1 else float(proba[0])
    outcome         = "CKD detected" if prediction == 1 else "No CKD detected"

    # 3 — SHAP local
    shap_vals = extract_shap(df_input)

    pairs = sorted(
        zip(FEATURE_ORDER, [float(v) for v in shap_vals], [float(v) for v in df_input.iloc[0].values.tolist()]),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]

    shap_lines = [
        f"- {f}: value={v:.2f}, contribution={s:+.3f} "
        f"({'higher' if s > 0 else 'lower'} risk)"
        for f, s, v in pairs
    ]

    # 4 — LIME local
    lime_pairs = extract_lime(df_input, top_n=5)
    lime_local = build_lime_local_summary(lime_pairs, outcome, confidence)

    # 5 — SHAP global (run on a small sample for speed — use df_input as proxy)
    # Ideally pass your full X_test here; for the API we use the single input
    shap_global = build_shap_global_summary(
        shap_vals.reshape(1, -1)   # single sample fallback
    )

    # 6 — Build prompt and call Claude
    if view == ViewMode.patient:
        prompt = build_patient_prompt(
            outcome, confidence, ckd_probability, pairs, shap_lines
        )
        message = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 400,
            messages   = [{"role": "user", "content": prompt}]
        )

    elif view == ViewMode.clinic:
        prompt = build_clinic_prompt(
            outcome, confidence, ckd_probability,
            pairs, shap_lines, lime_local, shap_global
        )
        message = client.messages.create(
            model      = "claude-sonnet-4-20250514",
            max_tokens = 2000,
            messages   = [{"role": "user", "content": prompt}],
            mcp_servers = [
                {
                    "type": "url",
                    "url" : "https://mcp.healthcare-server-url.com/sse",
                    "name": "healthcare-mcp"
                }
            ]
        )

    # 7 — Extract text from response (handles MCP multi-block responses)
    explanation = " ".join([
        block.text for block in message.content
        if hasattr(block, "text")
    ])

    return {
        "prediction"        : outcome,
        "confidence"        : round(confidence, 4),
        "ckd_probability"   : round(ckd_probability, 4),
        "view_mode"         : view,
        "shap_contributions": {f: round(s, 4) for f, s, _ in pairs},
        "lime_contributions": {cond: round(w, 4) for cond, w in lime_pairs},
        "explanation"       : explanation
    }