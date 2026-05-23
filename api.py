import os
import asyncio
import traceback
import joblib
import numpy as np
import pandas as pd
import shap
import anthropic

from enum import Enum
from pathlib import Path
from string import Template
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
from dotenv import load_dotenv
from lime.lime_tabular import LimeTabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from ucimlrepo import fetch_ucirepo
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="CKD Risk Prediction API",
    description="Cloud-deployed explainable ML system for early CKD detection",
    version="1.0.0"
)

# ── Environment ───────────────────────────────────────────────────────────────

# load .env first, then fall back to the named key file if it exists
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).resolve().parent / "ANTHROPIC_API_KEY.env", override=False)

anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
if not anthropic_api_key:
    raise RuntimeError(
        "ANTHROPIC_API_KEY not found. Ensure .env or ANTHROPIC_API_KEY.env "
        "is present and contains ANTHROPIC_API_KEY."
    )

client = anthropic.Anthropic(api_key=anthropic_api_key)

# ── Feature order ─────────────────────────────────────────────────────────────

# this has to match exactly what the model was trained on — don't reorder
FEATURE_ORDER = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
    'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc',
    'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]

# ── Load model artifacts ──────────────────────────────────────────────────────

# the trained random forest model and its median values for imputation
# both saved from the notebook — if these files are missing the API won't start
model    = joblib.load("ckd_model.pkl")
medians  = joblib.load("feature_medians.pkl")

# SHAP explainer wraps the model — TreeExplainer is the fast version for tree models
explainer = shap.TreeExplainer(model)

# ── Precompute global SHAP summary ────────────────────────────────────────────

# we saved shap_values_test.npy from the notebook — it's the SHAP values for
# all 80 test patients. we compute the global summary once at startup so every
# request can use it without recomputing
def build_shap_global_summary(shap_values_all: np.ndarray, top_n: int = 10) -> str:
    arr      = shap_values_all.reshape(1, -1) if shap_values_all.ndim == 1 else shap_values_all
    mean_abs = np.abs(arr).mean(axis=0)
    ranked   = sorted(zip(FEATURE_ORDER, mean_abs), key=lambda x: x[1], reverse=True)[:top_n]

    lines = ["Global feature importance (averaged across all test patients):"]
    for i, (feat, imp) in enumerate(ranked, 1):
        lines.append(f"  {i}. {feat}: mean |SHAP| = {float(imp):.4f}")

    lines.append("\nDirection of top 3 features:")
    for feat, _ in ranked[:3]:
        idx      = FEATURE_ORDER.index(feat)
        mean_val = float(arr[:, idx].mean())
        direction = "generally increases CKD risk" if mean_val > 0 else "generally decreases CKD risk"
        lines.append(f"  - {feat}: {direction} (mean SHAP = {mean_val:.4f})")

    return "\n".join(lines)

SHAP_VALUES_TEST    = np.load("shap_values_test.npy")       # shape (80, 24)
SHAP_GLOBAL_SUMMARY = build_shap_global_summary(SHAP_VALUES_TEST)

# ── Build LIME explainer ──────────────────────────────────────────────────────

# LIME needs to know the training data distribution so it can generate
# realistic perturbations around a test point — so we re-run the same
# preprocessing pipeline from the notebook to get X_train back
def build_lime_explainer() -> LimeTabularExplainer:
    X_train = pd.read_csv("train_data_X.csv")
    return LimeTabularExplainer(
        training_data = X_train.values,
        feature_names = FEATURE_ORDER,
        class_names   = ['notckd', 'ckd'],
        mode          = 'classification'
    )
# Build LIME explainer lazily to avoid failing at import time when network is unavailable.
lime_explainer = None
lime_build_error = None

def get_lime_explainer() -> Optional[LimeTabularExplainer]:
    """Return a cached LIME explainer, building it on first use.

    If building fails (network, SSL, etc.) we cache the error and return None.
    """
    global lime_explainer, lime_build_error
    if lime_explainer is not None:
        return lime_explainer
    try:
        lime_explainer = build_lime_explainer()
        lime_build_error = None
    except Exception as e:
        lime_build_error = str(e)
        print(f"Failed to build LIME explainer: {e}")
        return None
    return lime_explainer

# ── Prompt templates ──────────────────────────────────────────────────────────

PROMPT_FILE = Path(__file__).resolve().parent / "prompts.txt"

# prompts.txt uses [patient] and [clinic] section headers
# each section is a Python string.Template with $variable placeholders
def load_prompt_templates() -> dict:
    raw_text     = PROMPT_FILE.read_text(encoding="utf-8")
    templates    = {}
    current_name = None
    current_lines = []

    for line in raw_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if current_name:
                templates[current_name] = "\n".join(current_lines).strip()
            current_name  = stripped.strip("[]").lower()
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
        raise ValueError(
            f"Prompt template '{template_name}' not found in {PROMPT_FILE}"
        )
    return Template(template_text).safe_substitute(**kwargs)

# ── Enums and schemas ─────────────────────────────────────────────────────────

class ReportView(str, Enum):
    patient = "patient"
    clinic  = "clinic"

class PatientData(BaseModel):
    # these five are mandatory — the model leans on them heavily
    # (hemo and sg are the two strongest predictors)
    hemo: float = Field(..., description="Haemoglobin level (g/dL)")
    sg:   float = Field(..., description="Specific gravity")
    sc:   float = Field(..., description="Serum creatinine (mg/dL)")
    al:   float = Field(..., description="Albumin (0-5 scale)")
    pcv:  float = Field(..., description="Packed cell volume (%)")

    # worth providing if available — these come up in the model fairly often
    age:  Optional[float] = Field(None, description="Age in years")
    bp:   Optional[float] = Field(None, description="Blood pressure (mm/Hg)")
    bgr:  Optional[float] = Field(None, description="Blood glucose random (mg/dL)")
    bu:   Optional[float] = Field(None, description="Blood urea (mg/dL)")
    sod:  Optional[float] = Field(None, description="Sodium (mEq/L)")
    htn:  Optional[int]   = Field(None, description="Hypertension (0=no, 1=yes)")
    dm:   Optional[int]   = Field(None, description="Diabetes mellitus (0=no, 1=yes)")

    # less critical but included for completeness
    su:    Optional[float] = Field(None, description="Sugar (0-5 scale)")
    rbc:   Optional[int]   = Field(None, description="Red blood cells in urine (0=normal, 1=abnormal)")
    pc:    Optional[int]   = Field(None, description="Pus cells (0=normal, 1=abnormal)")
    pcc:   Optional[int]   = Field(None, description="Pus cell clumps (0=notpresent, 1=present)")
    ba:    Optional[int]   = Field(None, description="Bacteria (0=notpresent, 1=present)")
    pot:   Optional[float] = Field(None, description="Potassium (mEq/L)")
    wc:  Optional[float] = Field(None, description="White blood cell count (cells/cumm)")
    rc:  Optional[float] = Field(None, description="Red blood cell count (millions/cmm)")
    cad:   Optional[int]   = Field(None, description="Coronary artery disease (0=no, 1=yes)")
    appet: Optional[int]   = Field(None, description="Appetite (0=good, 1=poor)")
    pe:    Optional[int]   = Field(None, description="Pedal edema (0=no, 1=yes)")
    ane:   Optional[int]   = Field(None, description="Anaemia (0=no, 1=yes)")

    model_config = {"json_schema_extra": {"example": {
        "hemo": 11.2, "sg": 1.015, "sc": 1.2, "al": 1.0, "pcv": 38.0
    }}}

# ── Helpers ───────────────────────────────────────────────────────────────────

def impute_missing(patient_dict: dict) -> dict:
    # fill any None values with the median from training
    # medians is loaded once at startup so no file I/O here
    for feat in FEATURE_ORDER:
        if patient_dict.get(feat) is None:
            patient_dict[feat] = medians[feat]
    return patient_dict


def extract_shap(df_input: pd.DataFrame) -> np.ndarray:
    raw = explainer.shap_values(df_input)

    if isinstance(raw, list):
        # random forest returns a list — one array per class
        # index 1 is the CKD class, which is what we care about
        vals = np.array(raw[1]).flatten()
    else:
        # xgboost returns a single array
        vals = np.array(raw).flatten()

    # slice to FEATURE_ORDER length in case of any extra columns
    return vals[:len(FEATURE_ORDER)]


def _predict_proba(X: np.ndarray) -> np.ndarray:
    # LIME passes raw numpy arrays; wrap in a DataFrame so sklearn
    # doesn't warn about missing feature names
    return model.predict_proba(pd.DataFrame(X, columns=FEATURE_ORDER))


def extract_lime(df_input: pd.DataFrame, top_n: int = 5) -> list:
    expl = get_lime_explainer()
    if expl is None:
        # LIME explainer couldn't be built (network or dependency issue).
        # Return empty explanation so the API can still operate.
        return []

    exp = expl.explain_instance(
        df_input.iloc[0].values,
        _predict_proba,
        num_features=top_n
    )
    return exp.as_list()


def build_lime_local_summary(lime_pairs: list, outcome: str, confidence: float) -> str:
    # format the LIME output into a readable string for the prompt
    lines = [
        "Local explanation (specific to this patient):",
        f"  Prediction : {outcome} (confidence: {confidence:.1%})",
        "  Feature conditions that influenced this prediction:"
    ]
    for condition, weight in lime_pairs:
        direction = "toward CKD" if weight > 0 else "away from CKD"
        lines.append(f"  - {condition}: weight = {weight:.4f} ({direction})")
    return "\n".join(lines)

# ── Prompt builders ───────────────────────────────────────────────────────────

def build_patient_prompt(outcome: str, confidence: float, ckd_probability: float,
                         pairs: list, shap_lines: list) -> str:
    top_features  = [f for f, s, v in pairs[:3]]
    plain_outcome = (
        "your kidneys may be showing signs of chronic kidney disease"
        if "CKD detected" in outcome
        else "your kidneys appear to be healthy"
    )

    # variable names here must match the $placeholders in prompts.txt [patient] section
    return render_prompt(
        "patient",
        plain_outcome = plain_outcome,
        confidence    = f"{confidence:.1%}",
        top_features  = ", ".join(top_features),
        shap_lines    = "\n".join(shap_lines)
    )


def build_clinic_prompt(outcome: str, confidence: float, ckd_probability: float,
                        pairs: list, shap_lines: list,
                        lime_local: str, shap_global: str) -> str:
    feature_lookup_list = "\n".join([f"- {f}" for f, s, v in pairs])
    patient_values = "\n".join([
        f"- {f}: {float(v):.2f} (contribution: {float(s):+.3f})"
        for f, s, v in pairs
    ])

    return render_prompt(
        "clinic",
        outcome         = outcome,
        confidence      = f"{confidence:.1%}",
        ckd_probability = f"{ckd_probability:.1%}",
        feature_lookup_list = feature_lookup_list,
        patient_values  = patient_values,
        lime_local      = lime_local,
        shap_global     = shap_global
    )


async def _create_message(**kwargs):
    for attempt in range(4):
        try:
            return await asyncio.to_thread(client.messages.create, **kwargs)
        except Exception as e:
            # Some versions of the Anthropic client expose an OverloadedError
            # class; others do not. Avoid referencing a missing attribute and
            # instead detect transient overload/network errors heuristically.
            msg = str(e).lower()
            status = getattr(e, 'status_code', None)
            is_transient = (
                'overload' in msg or 'overloaded' in msg or
                '503' in msg or status in (502, 503, 504)
            )
            if is_transient and attempt < 3:
                await asyncio.sleep(2 ** attempt)  # 1 s, 2 s, 4 s
                continue
            raise


async def _call_claude_no_tools(prompt_text: str) -> tuple[str, str]:
    response = await _create_message(
        model      = "claude-sonnet-4-6",
        max_tokens = 16000,
        thinking   = {"type": "enabled", "budget_tokens": 10000},
        messages   = [{"role": "user", "content": prompt_text}]
    )
    thinking_text    = ""
    explanation_text = ""
    for block in response.content:
        if block.type == "thinking":
            thinking_text    += block.thinking
        elif block.type == "text":
            explanation_text += block.text
    return explanation_text, thinking_text


async def generate_clinic_report(prompt_text: str) -> tuple[str, str]:
    try:
        server_params = StdioServerParameters(command="healthcare-mcp", args=[])
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools_response = await session.list_tools()
                tools = [
                    {
                        "name"        : t.name,
                        "description" : t.description or "",
                        "input_schema": t.inputSchema or {"type": "object", "properties": {}}
                    }
                    for t in tools_response.tools
                ]

                messages      = [{"role": "user", "content": prompt_text}]
                thinking_text = ""

                while True:
                    response = await _create_message(
                        model      = "claude-sonnet-4-6",
                        max_tokens = 16000,
                        thinking   = {"type": "enabled", "budget_tokens": 10000},
                        tools      = tools,
                        messages   = messages
                    )

                    for block in response.content:
                        if block.type == "thinking":
                            thinking_text += block.thinking

                    if response.stop_reason != "tool_use":
                        explanation = "".join(
                            b.text for b in response.content if b.type == "text"
                        )
                        return explanation, thinking_text

                    tool_results = []
                    for block in response.content:
                        if block.type == "tool_use":
                            try:
                                result      = await session.call_tool(block.name, block.input)
                                content_str = "\n".join(
                                    c.text if hasattr(c, "text") else str(c)
                                    for c in result.content
                                )
                            except Exception as exc:
                                content_str = f"Tool error: {exc}"

                            tool_results.append({
                                "type"        : "tool_result",
                                "tool_use_id" : block.id,
                                "content"     : content_str
                            })

                    messages.append({"role": "assistant", "content": response.content})
                    messages.append({"role": "user",      "content": tool_results})

    except FileNotFoundError:
        # healthcare-mcp not installed on this host — fall back to direct Claude call
        return await _call_claude_no_tools(prompt_text)

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/")
def root():
    return {
        "name"   : "CKD Risk Prediction API",
        "version": "1.0.0",
        "status" : "running",
        "docs"   : "/docs",
        "health" : "/health",
        "predict": "/predict"
    }


@app.get("/health")
def health():
    return {
        "status"        : "ok",
        "api_key_loaded": anthropic_api_key is not None
    }


@app.post("/predict")
async def predict(patient: PatientData, view: ReportView = ReportView.patient):
    try:
        # 1 — impute missing optional fields using training medians
        data     = patient.model_dump()
        data     = impute_missing(data)
        df_input = pd.DataFrame([data])[FEATURE_ORDER]

        # 2 — run the model
        prediction      = int(model.predict(df_input)[0])
        proba           = model.predict_proba(df_input)[0]
        ckd_probability = float(proba[1])
        outcome         = "CKD detected" if prediction == 1 else "No CKD detected"
        confidence      = ckd_probability if prediction == 1 else float(proba[0])

        # 3 — SHAP local explanation for this patient
        # note: global summary is precomputed at startup as SHAP_GLOBAL_SUMMARY
        # this per-request call gives us the local contribution of each feature
        # for this specific patient, which goes into the pairs/shap_lines below
        shap_vals_raw = explainer.shap_values(df_input)
        if isinstance(shap_vals_raw, list):
            shap_vals = np.array(shap_vals_raw[1]).flatten()[:len(FEATURE_ORDER)]
        else:
            shap_vals = np.array(shap_vals_raw).flatten()[:len(FEATURE_ORDER)]

        feature_list = list(FEATURE_ORDER)
        shap_list    = [float(v) for v in shap_vals]
        input_list   = [float(v) for v in df_input.iloc[0].values]

        # top 5 features by absolute SHAP contribution for this patient
        pairs = sorted(
            zip(feature_list, shap_list, input_list),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]

        shap_lines = [
            f"- {f}: value={float(v):.2f}, contribution={float(s):.3f} "
            f"({'higher' if float(s) > 0 else 'lower'} risk)"
            for f, s, v in pairs
        ]

        # 4 — LIME local explanation
        # this is purely local — explains why the model made this specific
        # decision for this patient, independent of all other patients
        lime_pairs = extract_lime(df_input, top_n=5)
        lime_local = build_lime_local_summary(lime_pairs, outcome, confidence)

        # 5 — build prompt and call Claude
        # patient view: plain language, no thinking
        # clinic view: structured clinical report, extended thinking enabled
        if view == ReportView.patient:
            prompt_text = build_patient_prompt(
                outcome         = outcome,
                confidence      = confidence,
                ckd_probability = ckd_probability,
                pairs           = pairs,
                shap_lines      = shap_lines
            )
            response         = await _create_message(
                model      = "claude-haiku-4-5-20251001",
                max_tokens = 400,
                messages   = [{"role": "user", "content": prompt_text}]
            )
            explanation_text = "".join(
                block.text for block in response.content
                if getattr(block, "type", None) == "text"
            )
            thinking_text = "".join(
                block.thinking for block in response.content
                if getattr(block, "type", None) == "thinking"
            )
            if not explanation_text:
                raise RuntimeError(
                    "Claude response contained no text blocks; full response was: "
                    f"{response.content}"
                )

        else:  # clinic
            prompt_text = build_clinic_prompt(
                outcome         = outcome,
                confidence      = confidence,
                ckd_probability = ckd_probability,
                pairs           = pairs,
                shap_lines      = shap_lines,
                lime_local      = lime_local,
                shap_global     = SHAP_GLOBAL_SUMMARY
            )
            explanation_text, thinking_text = await generate_clinic_report(prompt_text)

        return {
            "prediction"        : outcome,
            "confidence"        : round(confidence, 4),
            "ckd_probability"   : round(ckd_probability, 4),
            "view"              : view.value,
            "shap_contributions": {f: round(float(s), 4) for f, s, _ in pairs},
            "lime_contributions": {cond: round(float(w), 4) for cond, w in lime_pairs},
            "explanation"       : explanation_text,
            "thinking"          : thinking_text   # empty string for patient view
        }

    except Exception as exc:
        # print the full traceback to the uvicorn terminal for debugging
        print(traceback.format_exc())
        detail = str(exc)
        if exc.__cause__ is not None:
            detail += f" | cause: {exc.__cause__}"
        elif exc.__context__ is not None:
            detail += f" | context: {exc.__context__}"
        return JSONResponse(
            status_code=500,
            content={"error": "Prediction failed", "detail": detail}
        )
