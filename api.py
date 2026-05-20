from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Optional
import joblib
import numpy as np
import pandas as pd
import shap
import anthropic
from dotenv import load_dotenv
import os

app = FastAPI(
    title="CKD Risk Prediction API",
    description="Cloud-deployed explainable ML system for early CKD detection",
    version="1.0.0"
)

# Load saved model and explainer
model = joblib.load("ckd_model.pkl")
explainer = shap.TreeExplainer(model)
load_dotenv(dotenv_path="ANTHROPIC-API-KEY.env") #calling Claude API
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Feature order must match exactly what the model was trained on
FEATURE_ORDER = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
    'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wbcc',
    'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]

# Define the input schema
class PatientData(BaseModel):
    # Mandatory fields
    hemo: float = Field(..., description="Haemoglobin level (g/dL)", example=11.2)
    sg: float = Field(..., description="Specific gravity", example=1.015)
    sc: float = Field(..., description="Serum creatinine (mg/dL)", example=1.2)
    al: float = Field(..., description="Albumin (0-5 scale)", example=1.0)
    pcv: float = Field(..., description="Packed cell volume (%)", example=38.0)

    # Strongly recommended
    age: Optional[float] = Field(None, description="Age in years")
    bp: Optional[float] = Field(None, description="Blood pressure (mm/Hg)")
    bgr: Optional[float] = Field(None, description="Blood glucose random (mg/dL)")
    bu: Optional[float] = Field(None, description="Blood urea (mg/dL)")
    sod: Optional[float] = Field(None, description="Sodium (mEq/L)")
    htn: Optional[int] = Field(None, description="Hypertension (0=no, 1=yes)")
    dm: Optional[int] = Field(None, description="Diabetes mellitus (0=no, 1=yes)")

    # Optional
    su: Optional[float] = Field(None, description="Sugar (0-5 scale)")
    rbc: Optional[int] = Field(None, description="Red blood cells in urine (0=normal, 1=abnormal)")
    pc: Optional[int] = Field(None, description="Pus cells (0=normal, 1=abnormal)")
    pcc: Optional[int] = Field(None, description="Pus cell clumps (0=notpresent, 1=present)")
    ba: Optional[int] = Field(None, description="Bacteria (0=notpresent, 1=present)")
    pot: Optional[float] = Field(None, description="Potassium (mEq/L)")
    wbcc: Optional[float] = Field(None, description="White blood cell count (cells/cumm)")
    rbcc: Optional[float] = Field(None, description="Red blood cell count (millions/cmm)")
    cad: Optional[int] = Field(None, description="Coronary artery disease (0=no, 1=yes)")
    appet: Optional[int] = Field(None, description="Appetite (0=good, 1=poor)")
    pe: Optional[int] = Field(None, description="Pedal edema (0=no, 1=yes)")
    ane: Optional[int] = Field(None, description="Anaemia (0=no, 1=yes)")

def impute_missing(patient_dict):
    """Fill missing optional fields with column medians from training data."""
    # Load saved training medians (save these from your notebook)
    medians = joblib.load("feature_medians.pkl")
    
    for feat in FEATURE_ORDER:
        if patient_dict.get(feat) is None:
            patient_dict[feat] = medians[feat]
    
    return patient_dict

@app.post("/predict")
def predict(patient: PatientData):
    # Convert to dict and impute missing values
    data = patient.dict()
    data = impute_missing(data)

    # Build dataframe in correct feature order
    df_input = pd.DataFrame([data])[FEATURE_ORDER]

    # Predict
    prediction = int(model.predict(df_input)[0])        # force plain Python int
    proba = model.predict_proba(df_input)[0]            # [prob_class_0, prob_class_1]
    ckd_probability = float(proba[1])                   # always index 1 for CKD

    outcome = "CKD detected" if prediction == 1 else "No CKD detected"
    confidence = ckd_probability if prediction == 1 else float(proba[0])

    # Force completely flat regardless of what SHAP returns
    shap_vals_raw = explainer.shap_values(df_input)
    shap_vals = np.array(shap_vals_raw).flatten()
    
    # Pick the right 24 values depending on model type
    if isinstance(shap_vals_raw, list):
        # RF returns shape (2, 1, 24) — take second class (CKD)
        shap_vals = np.array(shap_vals_raw[1]).flatten()[:len(FEATURE_ORDER)]
    else:
        # XGB returns shape (1, 24)
        shap_vals = np.array(shap_vals_raw).flatten()[:len(FEATURE_ORDER)]
    
    # Verify before sorting
    print("final shap_vals shape:", shap_vals.shape)  # must be (24,)
    print("sample value:", shap_vals[0], type(shap_vals[0]))
    
    feature_list = list(FEATURE_ORDER)
    shap_list = [float(v) for v in shap_vals]          # explicitly cast every element
    input_list = [float(v) for v in df_input.iloc[0].values]
    
    pairs = sorted(
        zip(feature_list, shap_list, input_list),
        key=lambda x: abs(x[1]),                        # x[1] is now guaranteed float
        reverse=True
    )[:5]

    shap_lines = [
    f"- {f}: value={float(v):.2f}, contribution={float(s):.3f} "
    f"({'higher' if float(s) > 0 else 'lower'} risk)"
    for f, s, v in pairs
    ]

    # Claude NL explanation
    prompt = f"""You are a senior nephrologist reviewing an AI CKD risk assessment.
Patient prediction: {outcome} (confidence: {confidence:.1%})
Top contributing factors:
{chr(10).join(shap_lines)}
Write a 3-4 sentence clinical summary in plain English. No ML terminology."""

    message = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    return {
        "prediction": outcome,
        "confidence": round(confidence, 4),
        "ckd_probability": round(ckd_probability, 4),
        "shap_contributions": {
            f: round(float(s), 4)
            for f, s, _ in pairs
        },
        "explanation": message.content[0].text
    }

@app.get("/health")
def health():
    return {"status": "ok"}