"""
vitaldiagnostique.py
--------------------
Streamlit interface for pediatric pneumonia prediction.
Calls whyvitals.py for medical explanation of model output.

Run with:
    streamlit run vitaldiagnostique.py

Feature order expected by Gradient_Boost.pkl (confirmed from feature_importances_):
    Index 0  Gender                 0.1%
    Index 1  Age                    3.2%
    Index 2  Cough                  3.4%
    Index 3  Fever                 12.9%
    Index 4  Shortness_of_breath    0.6%
    Index 5  Chest_pain             4.9%
    Index 6  Confusion             64.8%
    Index 7  Fatigue                0.2%
    Index 8  Oxygen_saturation      0.3%
    Index 9  Crackles               0.0%
    Index 10 Sputum_color           0.4%
    Index 11 Temperature            9.2%
"""

import warnings
import os
import sys

import joblib
import numpy as np
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from whyvitals import VitalInput, explain

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pediatric Pneumonia Diagnostic",
    page_icon=":material/monitor_heart:",
    layout="centered",
)

st.markdown(
    '<link rel="stylesheet" '
    'href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
#MainMenu, footer, header {visibility: hidden;}
[data-testid="stSidebar"]        {display: none;}
[data-testid="collapsedControl"] {display: none;}

.stApp           { background-color: #ffffff; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }

.page-title {
    font-size: 1.7rem; font-weight: 700; color: #000000;
    margin-bottom: 0.3rem; padding: 0.6rem 0;
}
.page-caption {
    font-size: 0.9rem; color: #000000; margin-bottom: 1.5rem;
}
.section-title {
    font-size: 1.1rem; font-weight: 600; color: #000000;
    border-left: 4px solid #1e90ff; padding-left: 0.6rem;
    margin: 1.8rem 0 1rem 0;
    background-color: #ffffff; border-radius: 0 6px 6px 0;
}
.divider {
    height: 2px;
    background: linear-gradient(90deg, #1e90ff, #ffffff);
    margin: 1rem 0 1.5rem 0; border-radius: 2px;
}
.tag-pill {
    background: #ffffff; color: #000000;
    border: 1.5px solid #000080;
    padding: 4px 12px; border-radius: 14px;
    font-size: 12px; display: inline-block;
    margin: 3px; font-weight: 500;
}
.warn-box {
    background: #fff8e1; border-left: 4px solid #f0a500;
    padding: 0.6rem 1rem; border-radius: 6px;
    font-size: 0.85rem; color: #000000; margin-bottom: 1rem;
}
.verdict-sick {
    background: #fde8e8; border-left: 5px solid #cc0000;
    padding: 0.8rem 1.2rem; border-radius: 6px;
    color: #000000; font-weight: 600; font-size: 1rem;
    margin-bottom: 1rem;
}
.verdict-ok {
    background: #e8f5e9; border-left: 5px solid #1e90ff;
    padding: 0.8rem 1.2rem; border-radius: 6px;
    color: #000000; font-weight: 600; font-size: 1rem;
    margin-bottom: 1rem;
}
label,
.stSelectbox label,
.stNumberInput label {
    color: #000000 !important; font-weight: 500 !important;
}
.stTextInput input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] * {
    color: #000000 !important; background-color: #ffffff !important;
}
div[data-baseweb="select"] > div {
    border-color: #1e90ff !important;
    border-radius: 8px !important;
    background-color: #ffffff !important;
}
.stNumberInput > div > div > input {
    border-color: #1e90ff !important; border-radius: 8px !important;
}
.stExpander {
    border: 1.5px solid #000080 !important;
    border-radius: 10px !important;
    background-color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(
    "<div class='page-title'>"
    "<i class='fas fa-heartbeat' style='color:#1e90ff;margin-right:0.5rem;'></i>"
    " Pediatric Pneumonia Diagnostic"
    "</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='page-caption'>"
    "Enter patient vital signs. The model predicts pneumonia risk and explains why."
    "</div>",
    unsafe_allow_html=True,
)
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model — hard stop if missing, version warning if sklearn mismatch
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), '..', '..', 'models', 'Gradient_Boost.pkl'
)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = joblib.load(MODEL_PATH)
    mismatch = any(
        "InconsistentVersionWarning" in str(w.category) for w in caught
    )
    return m, mismatch


try:
    model, version_mismatch = load_model()
except FileNotFoundError:
    st.markdown(
        f"<div class='warn-box'>"
        f"<i class='fas fa-circle-xmark' style='color:#cc0000;margin-right:6px;'></i>"
        f"<strong>Model not found.</strong> Expected: <code>{MODEL_PATH}</code><br>"
        f"Place <code>Gradient_Boost.pkl</code> in the <code>models/</code> folder and restart."
        f"</div>",
        unsafe_allow_html=True,
    )
    st.stop()

# sklearn version mismatch — visible warning, non-blocking
if version_mismatch:
    import sklearn as _sklearn
    st.markdown(
        "<div class='warn-box'>"
        "<i class='fas fa-triangle-exclamation' style='color:#f0a500;margin-right:6px;'></i>"
        f"<strong>sklearn version mismatch:</strong> model trained on 1.6.1, "
        f"running {_sklearn.__version__}. Predictions may be unreliable. "
        "Retrain and re-export the model with your current version."
        "</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title'>Patient Vital Signs</div>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    gender     = st.selectbox("Gender",                ["M", "F"])
    age        = st.number_input("Age (years)",        min_value=1, max_value=16, value=5, step=1)
    cough      = st.selectbox("Cough",                 ["Dry", "Wet", "Bloody"])
    fever      = st.selectbox("Fever",                 ["Low", "Moderate", "High"])
    sob        = st.selectbox("Shortness of breath",   ["Mild", "Moderate", "Severe"])
    chest_pain = st.selectbox("Chest pain",            ["Mild", "Moderate", "Severe"])

with col2:
    fatigue      = st.selectbox("Fatigue",             ["Mild", "Moderate", "Severe"])
    confusion    = st.selectbox("Confusion",           ["No", "Yes"])
    spo2         = st.number_input("SpO₂ (%)",         min_value=85.0, max_value=100.0, value=97.0, step=0.5)
    crackles     = st.selectbox("Crackles",            ["No", "Yes"])
    sputum_color = st.selectbox("Sputum color",        ["None", "Clear", "Yellow", "Green", "Rust"])
    temperature  = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0, step=0.1)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Analyze button
# ---------------------------------------------------------------------------
if st.button(":material/biotech:  Analyze", use_container_width=True, type="primary"):

    GENDER_MAP = {"M": 1, "F": 0}
    COUGH_MAP  = {"Dry": 0, "Wet": 1, "Bloody": 2}
    FEVER_MAP  = {"Low": 0, "Moderate": 1, "High": 2}
    SOB_MAP    = {"Mild": 0, "Moderate": 1, "Severe": 2}
    CP_MAP     = {"Mild": 0, "Moderate": 1, "Severe": 2}
    CONF_MAP   = {"No": 0, "Yes": 1}
    FAT_MAP    = {"Mild": 0, "Moderate": 1, "Severe": 2}
    CRACK_MAP  = {"No": 0, "Yes": 1}
    SPUTUM_MAP = {"None": 0, "Clear": 1, "Yellow": 2, "Green": 3, "Rust": 4}

    feature_vector = np.array([[
        GENDER_MAP[gender],        # 0  Gender              0.1%
        int(age),                  # 1  Age                 3.2%
        COUGH_MAP[cough],          # 2  Cough               3.4%
        FEVER_MAP[fever],          # 3  Fever              12.9%
        SOB_MAP[sob],              # 4  Shortness_of_breath  0.6%
        CP_MAP[chest_pain],        # 5  Chest_pain           4.9%
        CONF_MAP[confusion],       # 6  Confusion           64.8%
        FAT_MAP[fatigue],          # 7  Fatigue              0.2%
        float(spo2),               # 8  Oxygen_saturation    0.3%
        CRACK_MAP[crackles],       # 9  Crackles             0.0%
        SPUTUM_MAP[sputum_color],  # 10 Sputum_color         0.4%
        float(temperature),        # 11 Temperature          9.2%
    ]])

    # --- Model prediction — no fallback, no override ---
    try:
        prediction = int(model.predict(feature_vector)[0])
    except Exception as e:
        st.markdown(
            f"<div class='warn-box'>"
            f"<i class='fas fa-circle-xmark' style='color:#cc0000;margin-right:6px;'></i>"
            f"<strong>Prediction failed:</strong> {e}"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.stop()

    try:
        proba      = float(model.predict_proba(feature_vector)[0][1])
        confidence = f"{proba * 100:.0f}%"
    except Exception:
        confidence = "N/A"

    # --- Explanation from whyvitals (does not alter model output) ---
    vitals = VitalInput(
        Gender=gender,
        Age=int(age),
        Cough=cough,
        Fever=fever,
        Shortness_of_breath=sob,
        Chest_pain=chest_pain,
        Fatigue=fatigue,
        Confusion=confusion,
        Oxygen_saturation=float(spo2),
        Crackles=crackles,
        Sputum_color=sputum_color,
        Temperature=float(temperature),
    )
    result = explain(vitals, prediction)

    # ---------------------------------------------------------------------------
    # Display — pure HTML so Font Awesome icons render correctly
    # ---------------------------------------------------------------------------
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    if prediction == 1:
        st.markdown(
            f"<div class='verdict-sick'>"
            f"<i class='fas fa-circle-exclamation' style='color:#cc0000;margin-right:8px;'></i>"
            f"Predicted: Sick — Pneumonia Likely &nbsp;&nbsp; confidence: {confidence}"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='verdict-ok'>"
            f"<i class='fas fa-circle-check' style='color:#1e90ff;margin-right:8px;'></i>"
            f"Predicted: Not Sick — Pneumonia Unlikely &nbsp;&nbsp; confidence: {confidence}"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown(result["summary"])

    if result["tags"]:
        st.markdown("**Abnormal signals detected:**")
        tag_cols = st.columns(min(3, len(result["tags"])))
        for i, tag in enumerate(result["tags"][:6]):
            tag_cols[i % 3].markdown(
                f"<span class='tag-pill'>{tag}</span>",
                unsafe_allow_html=True,
            )

    if result["interactions"]:
        st.markdown("**Key signal interactions:**")
        for note in result["interactions"]:
            st.markdown(f"> {note}")

    with st.expander("Why each vital sign matters"):
        for feat, info in result["feature_notes"].items():
            if info:
                st.markdown(f"**{feat.replace('_', ' ')}** — {info['note']}")
                st.caption(f"Ref: {info['ref']}")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

if st.button(":material/arrow_back:  Back to Home", use_container_width=False):
    st.switch_page("app.py")