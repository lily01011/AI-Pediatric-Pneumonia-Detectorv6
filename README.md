# 🫁 AI Pediatric Pneumonia Detector

> An intelligent clinical decision support system for detecting pediatric pneumonia from chest X-rays and vital signs, powered by DenseNet121 and Gradient Boosting — built for underserved healthcare settings.

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Objectives & Use Case](#objectives--use-case)
3. [System Architecture](#system-architecture)
4. [Database & Storage Design](#database--storage-design)
5. [Features & Functionalities](#features--functionalities)
6. [AI Models](#ai-models)
7. [Explainability Layer — Medical Dictionaries](#explainability-layer--medical-dictionaries)
8. [Technologies Used](#technologies-used)
9. [Project Structure](#project-structure)
10. [Installation & Setup](#installation--setup)
11. [Usage Guide](#usage-guide)
12. [Team](#team)
13. [Acknowledgements](#acknowledgements)
14. [Technical Notes & Known Issues](#technical-notes--known-issues)
15. [License](#license)

---

## Project Overview

**AI Pediatric Pneumonia Detector** is a full-stack clinical decision support application developed at the **University of Saida, Algeria (2026)**. It assists radiologists and pediatricians in diagnosing pneumonia in children under five by combining:

- **Deep Learning (DenseNet121)** for chest X-ray classification with **Grad-CAM** visual explainability
- **Gradient Boosting** for vital signs-based risk assessment
- A **medically-referenced explanation engine** (`whyvitals.py` + `whyxray.py`) that translates raw AI predictions into clinician-facing, citation-backed reasoning
- A **doctor-centric file management system** for organized, per-patient clinical records

The system is designed specifically for resource-constrained and underserved healthcare environments, where diagnostic delays directly impact child mortality outcomes.

---

## Objectives & Use Case

**Primary Objective:** Reduce diagnostic delays for pediatric pneumonia by providing AI-assisted screening tools to frontline medical practitioners.

**Target Users:** Pediatricians, radiologists, and general practitioners — especially in settings with limited specialist access.

**Core Use Cases:**

| Scenario | System Response |
|---|---|
| Doctor opens the app for the first time | Prompted to complete profile; data namespace is created |
| Doctor registers a new patient | Patient folder created; structured CSV + clinical PDF generated |
| Doctor uploads a chest X-ray | DenseNet121 classifies NORMAL / PNEUMONIA; Grad-CAM heatmap highlights focus regions |
| Doctor enters vital signs | Gradient Boosting model predicts risk class with confidence score |
| Doctor requests explanation | `whyvitals.py` / `whyxray.py` generates cited clinical rationale for the prediction |
| Doctor saves results | Diagnostic report (CSV + PDF) written to patient folder |
| Doctor plans treatment | Hospitalization or home treatment form; medication table + discharge plan |

---

## System Architecture

The application is organized as a multi-page **Streamlit** web application with a modular page structure and a shared file-based data layer.

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit Frontend                  │
│  app.py → profile.py → Add_Patient.py              │
│           Diagnostic.py → Treatment.py              │
│           About_Us.py                               │
└──────────────────┬──────────────────────────────────┘
                   │ function calls
┌──────────────────▼──────────────────────────────────┐
│               Application Logic Layer                │
│  profile_db.py      ← Doctor session management     │
│  patient_db.py      ← Patient CRUD + PDF/CSV gen    │
└──────────────────┬──────────────────────────────────┘
                   │ file I/O
┌──────────────────▼──────────────────────────────────┐
│            File-Based Storage Layer                  │
│  data/<Region-Hospital>/<DoctorName>/               │
│    └── <PatientID_Name>/                            │
│          ├── patient_data.csv / .pdf                │
│          ├── diagnostic_report.csv / .pdf           │
│          ├── xray_original.png                      │
│          ├── gradcam_heatmap.png                    │
│          └── gradcam_overlay.png                    │
└──────────────────┬──────────────────────────────────┘
                   │ model inference + explanation
┌──────────────────▼──────────────────────────────────┐
│          AI / ML Models + Explainability             │
│  densenet121_best_model.keras  ← X-ray CNN          │
│  Gradient_Boost.pkl            ← Vital signs model  │
│  gradcam.py                    ← Grad-CAM util      │
│  whyvitals.py                  ← Vital sign explainer│
│  whyxray.py                    ← X-ray explainer    │
└─────────────────────────────────────────────────────┘
```

**Key Architectural Decisions:**

- **No traditional database (SQL/NoSQL):** The system stores heterogeneous data (structured forms, free text, PDFs, images, AI outputs) in a hierarchical file system. This avoids BLOB storage overhead and schema complexity at the current application scale.
- **Doctor-scoped data isolation:** Each doctor's data is namespaced by `<Region>-<Hospital>/<DoctorName>/`, ensuring no cross-contamination between practitioners.
- **Dual-format outputs (CSV + PDF):** Every record is written in both a machine-readable format (for data pipelines) and a human-readable format (for clinical review and printing).
- **Session state via flat file:** The active doctor session is persisted to `apppy/.active_doctor` — a plain-text file containing the current doctor's folder path — enabling lightweight stateful navigation across Streamlit page reloads.
- **Decoupled explanation layer:** The `whyvitals.py` and `whyxray.py` modules are pure dictionaries with no Streamlit dependencies, making them independently testable and reusable in any Python context (API, CLI, unit tests).

---

## Database & Storage Design

The system uses a **file-based hierarchical storage architecture**. There is no SQL or NoSQL engine. All data persists under the `data/` root directory.

### Full Directory Tree

```
ai-pediatric-pneumonia-detector/
│
└── data/
    └── <Region + Hospital Name>/          ← Created when doctor saves profile
        │   Example: saida-AhmedMedagriCentralHospital/
        │
        └── <DoctorLastName-DoctorFirstName>/   ← Doctor namespace
            │
            ├── doctor_profile.csv              ← Doctor credentials (CSV)
            │
            └── <PatientID>_<FirstName>_<LastName>/   ← Created at patient save
                │   Example: 1001_Aya_Benali/
                │
                ├── medical_history_pdf/         ← Uploaded PDFs (may be empty)
                ├── family_history_pdf/           ← Uploaded PDFs (may be empty)
                ├── clinical_notes_pdf/           ← Uploaded PDFs (may be empty)
                ├── xray/                         ← Uploaded raw X-ray images
                │
                ├── 1001_Aya_Benali.csv          ← Structured patient record
                ├── 1001_Aya_Benali.pdf          ← Formatted patient record (clinical)
                │
                ├── xray_original.png            ← Saved after diagnostic
                ├── gradcam_heatmap.png          ← Grad-CAM output
                ├── gradcam_overlay.png          ← Overlay visualization
                ├── diagnostic_report.csv        ← Full diagnostic data
                └── diagnostic_report.pdf        ← Full diagnostic report (clinical)
```

### Data Entities

| Entity | Storage Format | Purpose |
|---|---|---|
| Doctor Profile | `doctor_profile.csv` | Credentials, institution, specialization |
| Active Session | `apppy/.active_doctor` (plain text) | Points to active doctor folder path |
| Patient Record | `<id>_name.csv` + `<id>_name.pdf` | Demographics, contact, clinical notes |
| Medical Uploads | PDFs in categorized subfolders | Scanned historical / family records |
| Diagnostic Report | `diagnostic_report.csv` + `.pdf` | Vital signs, X-ray result, Grad-CAM |
| Grad-CAM Outputs | `.png` images | AI visual explanation for clinicians |

### Data Isolation

Patient lookups on the Diagnostic page are **scoped to the currently active doctor's folder only**. The application does not traverse sibling doctor directories, ensuring full data separation between practitioners.

---

## Features & Functionalities

### 👤 Doctor Profile Management
- First-launch detection with mandatory profile setup gate
- Professional credential form (name, degree, specialization, hospital, region, country)
- Automatic creation of doctor-namespaced directory on save
- Profile data persisted to `doctor_profile.csv`

### 🧒 Patient Registration (Add Patient)
- Auto-incrementing patient ID generation (starting at 1001)
- Structured intake form: demographics, contact, date of birth, blood type
- Free-text clinical fields: medical history, family illnesses, clinical notes
- Optional PDF uploads per clinical category
- Simultaneous generation of CSV (machine-readable) and PDF (clinical) records on save

### 🔬 AI Diagnostic Engine (Two Pathways)

**Pathway A — Vital Signs Analysis:**
- Input: gender, age, cough severity, fever grade, shortness of breath, chest pain, fatigue, confusion, oxygen saturation, blood type
- Model: Gradient Boosting classifier (`Gradient_Boost.pkl`)
- Output: PNEUMONIA / NORMAL label + confidence percentage
- Explanation: `whyvitals.py` generates a feature-by-feature clinical rationale with medical references

**Pathway B — Chest X-Ray Analysis:**
- Input: uploaded chest X-ray (PNG, JPG, JPEG, DCM, BMP, TIFF)
- Model: Fine-tuned DenseNet121 (`densenet121_best_model.keras`)
- Output: PNEUMONIA / NORMAL label + confidence percentage
- Explainability: Grad-CAM heatmap and overlay visualization (saved as PNG)
- Explanation: `whyxray.py` maps the prediction outcome to a structured radiology dictionary with cited radiological signs

**Combined Diagnostic Report:**
- Both results consolidated into a single `diagnostic_report.pdf`
- Conflicting results flagged with recommendation for specialist review
- Disclaimer printed on every report: AI results must be reviewed by a qualified professional

### 💊 Treatment Planning
- Home plan: medication table, next appointment scheduling, emergency warning signs, emergency handover notes for night-shift physicians
- Treatment plan finalized and saved per patient

### ℹ️ About Us
- Static team information page (no database interaction)
- Displays team member cards: name, role, degree, institution, email

---

## AI Models

| Model | File | Task | Architecture |
|---|---|---|---|
| X-ray CNN | `densenet121_best_model.keras` | Binary classification (PNEUMONIA / NORMAL) | DenseNet121 fine-tuned |
| Vital Signs Classifier | `Gradient_Boost.pkl` | Binary classification from 12 vital sign features | Gradient Boosting (scikit-learn) |
| Explainability | `gradcam.py` | Heatmap generation over last convolutional layer | Grad-CAM (TF GradientTape) |
| Vital Explanation | `whyvitals.py` | Feature-level clinical notes + interaction reasoning | Medical dictionary (rule-based) |
| X-ray Explanation | `whyxray.py` | Prediction-level radiology notes with cited sources | Medical dictionary (rule-based) |

**X-ray Classification Threshold:** `pred_score > 0.260` → PNEUMONIA

**Grad-CAM Layer:** `conv5_block16_concat` (last dense block in DenseNet121)

**Vital Sign Feature Vector (12 dimensions):**

```
[Gender, Age, Cough, Fever, Shortness_of_breath,
 Chest_pain, Fatigue, Confusion, Oxygen_saturation,
 Crackles, Sputum_color, Temperature]
```

---

## Explainability Layer — Medical Dictionaries

One of the distinguishing features of **v6** is a fully medically-referenced **explanation layer** that accompanies every AI prediction. Rather than showing a raw probability score to the clinician, the system translates model outputs into structured, citation-backed clinical reasoning through two dedicated Python modules.

---

### 📄 `whyvitals.py` — Vital Signs Explanation Engine

**Location:** `apppy/whyvitals.py`

**Purpose:** Generates concise, medically-referenced textual explanations for the Gradient Boosting model's pneumonia prediction based on the 12 input vital sign features.

#### How It Works

The module exposes a single main entry point:

```python
from whyvitals import explain, VitalInput

vital = VitalInput(
    Gender="M", Age=4, Cough="Wet", Fever="High",
    Shortness_of_breath="Severe", Chest_pain="Moderate",
    Fatigue="Moderate", Confusion="No",
    Oxygen_saturation=91.0, Crackles="Yes",
    Sputum_color="Green", Temperature=39.2
)

result = explain(vital, prediction=1)
```

The returned dictionary contains:

| Key | Type | Description |
|---|---|---|
| `verdict` | `str` | `"Sick"` or `"Not Sick"` |
| `score` | `int` | Internal risk score (0–100), normalized from weighted feature signals |
| `summary` | `str` | One-sentence clinical summary with primary driver and signal count |
| `primary_driver` | `str` | Clinical note for the single highest-weight abnormal feature |
| `tags` | `list[str]` | All active abnormal signal labels (e.g., `"Localized crackles"`) |
| `interactions` | `list[str]` | Active multi-feature interaction notes with references |
| `feature_notes` | `dict` | Per-feature `{note, ref}` dictionary for all 12 inputs |

#### Feature Coverage

The `FEATURE_EXPLANATIONS` dictionary covers all 12 vital sign features, each mapped to clinically relevant value ranges:

| Feature | Value Bands | Top Reference |
|---|---|---|
| `Oxygen_saturation` | ≤88%, 89–92%, 93–94%, ≥95% | WHO IMCI 2014; Mandell et al., CID 2007 |
| `Crackles` | Yes / No | Wipf JE, Ann Intern Med 1999 |
| `Sputum_color` | Rust / Green / Yellow / Clear / None | Mandell et al., CID 2007 |
| `Temperature` | <38°C, 38–38.4, 38.5–38.9, 39–39.4, ≥39.5 | Fine MJ et al., NEJM 1997 |
| `Confusion` | Yes / No | Lim WS et al., Thorax 2009 (CURB-65) |
| `Shortness_of_breath` | Mild / Moderate / Severe | Mandell et al., CID 2007 |
| `Fever` | Low / Moderate / High | Lim WS et al., Thorax 2009 |
| `Cough` | Dry / Wet / Bloody | Metlay JP et al., AJRCCM 2019 |
| `Chest_pain` | Mild / Moderate / Severe | Mandell et al., CID 2007 |
| `Fatigue` | Mild / Moderate / Severe | Fine MJ et al., NEJM 1997 |
| `Age` | <2 yrs, 2–5, 6–12, 13–16 | WHO IMCI 2014; Rudan I, Lancet 2008 |
| `Gender` | M / F | Shah SN, StatPearls 2023 |

#### Interaction Detection

Beyond individual feature explanations, `whyvitals.py` detects **five clinically significant feature combinations** that carry a higher positive likelihood ratio than individual signals:

| Interaction | Clinical Significance |
|---|---|
| High fever + Rust/Green sputum | Pathognomonic pattern for bacterial CAP (S. pneumoniae) — Mandell et al., CID 2007 |
| SpO₂ ≤92% + Confusion | Cerebral hypoperfusion red flag; maps to CURB-65 "U+" — Lim WS et al., Thorax 2009 |
| Crackles + SpO₂ ≤94% | Focal consolidation with measurable gas exchange impairment — Wipf JE, Ann Intern Med 1999 |
| Age <5 + Severe dyspnea | WHO IMCI danger sign requiring hospital admission — WHO IMCI 2014 |
| Wet cough + Crackles + High fever | Classic triad for typical bacterial CAP (S. pneumoniae, H. influenzae) — Jain S et al., NEJM 2015 |

#### Score Weighting

The internal risk score uses evidence-based feature weights that mirror the trained model's feature importance:

```python
SCORE_WEIGHTS = {
    "Confusion":          28,
    "Oxygen_saturation":  30,
    "Crackles":           24,
    "Sputum_color":       22,
    "Temperature":        18,
    "Shortness_of_breath":18,
    "Age":                18,
    "Fever":              14,
    "Chest_pain":         12,
    "Cough":              10,
    "Fatigue":             7,
    "Gender":              2,
}
```

#### Medical References (`whyvitals.py`)

- Mandell LA et al. *IDSA/ATS Consensus Guidelines on CAP.* CID 2007;44:S27–S72.
- WHO. *IMCI Integrated Management of Childhood Illness.* 2014.
- Lim WS et al. *BTS Guidelines for CAP.* Thorax 2009;64(Suppl III).
- Fine MJ et al. *Pneumonia Patient Outcomes Research Team (PORT).* NEJM 1997.
- Shah SN et al. *Pediatric Pneumonia.* StatPearls 2023.
- Metlay JP et al. *ATS/IDSA Update on CAP.* AJRCCM 2019;200(7):e45–e67.
- Jain S et al. *Community-Acquired Pneumonia (EPIC study).* NEJM 2015.
- Rudan I et al. *Epidemiology of CAP in children.* Lancet 2008.
- Wipf JE et al. *Diagnosing Pneumonia by Physical Examination.* Ann Intern Med 1999.

---

### 📄 `whyxray.py` — X-Ray Prediction Explanation Dictionary

**Location:** `apppy/whyxray.py`

**Purpose:** Maps DenseNet121 prediction outcomes to structured, clinician-facing radiology explanations with cited radiological signs and relevant literature.

#### How It Works

The module exposes a single `EXPLANATION_DICT` dictionary, keyed by prediction outcome:

```python
from whyxray import EXPLANATION_DICT

# Access the explanation for a pneumonia prediction
explanation = EXPLANATION_DICT["pneumonia"]

# Access for a normal result
explanation = EXPLANATION_DICT["normal"]

# Access for a borderline/uncertain result
explanation = EXPLANATION_DICT["uncertain"]
```

Each entry contains:

| Key | Type | Description |
|---|---|---|
| `title` | `str` | Clinical heading for the result category |
| `summary` | `str` | Plain-language summary of what the model detected |
| `signs` | `list[dict]` | List of radiological signs, each with `name` and `description` |
| `clinical_note` | `str` | Contextual guidance for the clinician, including model performance metrics |
| `sources` | `list[dict]` | Cited references with `label` and `url` |

#### Prediction Categories

**`"pneumonia"`** — Radiological Signs Consistent with Pneumonia

Covers five classic imaging patterns the DenseNet121 model is trained to detect:

| Sign | Description |
|---|---|
| Lobar / Segmental Consolidation | Homogeneous opacification with air bronchograms — typical of bacterial CAP |
| Air Bronchograms | Radiolucent airways within consolidation — confirms airspace (not pleural) disease |
| Peribronchial Cuffing | Bronchial wall thickening — common in viral and Mycoplasma pneumonia in children |
| Reticulonodular / Interstitial Opacities | Bilateral diffuse haziness — typical of viral (RSV, influenza) or atypical pneumonia |
| Perihilar Haziness | Bilateral hilar opacity — associated with viral LRTI or atypical pathogens in young children |

> **Clinical note:** Threshold 0.260 optimized for ≥95% sensitivity per ROC analysis on the validation set.

**`"normal"`** — No Radiological Signs of Pneumonia Detected

Documents four reassuring normal chest X-ray findings:

| Sign | Description |
|---|---|
| Clear Lung Fields | No focal consolidation, interstitial opacities, or perihilar haziness |
| Normal Cardiothoracic Ratio | Cardiac width <50% thoracic diameter — excludes pulmonary edema mimic |
| Intact Costophrenic Angles | Sharp bilateral angles — no pleural effusion blunting |
| Normal Diaphragmatic Contour | Smooth bilateral domes without sub-phrenic air or infiltrate |

> **Clinical note:** Model sensitivity on internal test set: **95.01%**; on external validation: **97.21%**. A negative result lowers — but does not eliminate — pneumonia probability.

**`"uncertain"`** — Borderline Prediction — Manual Review Recommended

Triggered when the prediction score falls close to the decision threshold (0.260):

| Sign | Description |
|---|---|
| Ambiguous Opacities | Subtle density may represent early consolidation, atelectasis, or overlying soft tissue |
| Suboptimal Image Quality | Rotation, under/over-exposure, or patient movement can push predictions toward the threshold boundary |

> **Clinical note:** Request senior radiologist review. Consider lateral projection, repeat PA film in 24–48h, or chest HRCT if clinical suspicion remains.

#### Medical References (`whyxray.py`)

- WHO. *Pediatric Pneumonia Fact Sheet.* 2024.
- Rajpurkar P et al. *CheXNet: Radiologist-Level Pneumonia Detection.* Stanford AI Lab, 2017.
- Radiopaedia. *Pneumonia (Chest X-Ray Signs).* radiopaedia.org.
- RSNA. *Pediatric Chest X-Ray Interpretation.* rsna.org.
- Bramson RT et al. *Imaging of Community-Acquired Pneumonia in Children.* AJR 2005.
- Mollura DJ et al. *AI Detection of Pneumonia in Low-Resource Settings.* PLOS Digital Health, 2025.

---

### Integration Pattern

Both explanation modules are designed to integrate cleanly into the Diagnostic page (`Diagnostic.py`):

```python
# After vital signs prediction
from whyvitals import explain, VitalInput
vitals = VitalInput(...)                        # populate from form inputs
explanation = explain(vitals, prediction=model_output)
st.write(explanation["summary"])
for feat, note in explanation["feature_notes"].items():
    st.caption(f"**{feat}:** {note['note']} — *{note['ref']}*")

# After X-ray prediction
from whyxray import EXPLANATION_DICT
outcome = "pneumonia" if pred_score > 0.260 else "normal"
xray_exp = EXPLANATION_DICT[outcome]
st.subheader(xray_exp["title"])
st.write(xray_exp["summary"])
for sign in xray_exp["signs"]:
    st.markdown(f"- **{sign['name']}:** {sign['description']}")
```

---

## Technologies Used

| Layer | Technology |
|---|---|
| Frontend / UI | Streamlit |
| Deep Learning | TensorFlow / Keras |
| Machine Learning | scikit-learn, joblib |
| Image Processing | OpenCV, Pillow, NumPy, Matplotlib |
| PDF Generation | ReportLab |
| Data Storage | File system (CSV + PDF + PNG) |
| Explainability (X-ray) | `whyxray.py` — rule-based radiology dictionary |
| Explainability (Vitals) | `whyvitals.py` — weighted feature scoring + interaction engine |
| Language | Python 3.10+ |

---

## Project Structure

```
ai-pediatric-pneumonia-detector/
│
├── apppy/
│   ├── app.py                  ← Main entry point (Streamlit)
│   ├── profile_db.py           ← Doctor profile logic & session management
│   ├── patient_db.py           ← Patient CRUD, CSV/PDF generation
│   ├── whyvitals.py            ← Vital signs explanation engine (NEW v6)
│   ├── whyxray.py              ← X-ray prediction explanation dictionary (NEW v6)
│   ├── .active_doctor          ← Runtime session file (auto-generated)
│   │
│   └── pages/
│       ├── Add_Patient.py      ← Patient intake form
│       ├── Diagnostic.py       ← AI diagnostic engine (X-ray + vital signs)
│       ├── Treatment.py        ← Treatment plan builder
│       ├── profile.py          ← Doctor profile settings
│       └── About_Us.py         ← Team information page
│
├── models/
│   ├── densenet121_best_model.keras   ← CNN model (X-ray classification)
│   ├── Gradient_Boost.pkl             ← Vital signs classifier
│   └── gradcam.py                     ← Grad-CAM visualization utility
│
├── data/                        ← Auto-generated at runtime (see storage design)
│
└── assets/
    └── logo.png                 ← Application logo
```

---

## Installation & Setup

### Prerequisites

- Python 3.10 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/ai-pediatric-pneumonia-detector.git
cd ai-pediatric-pneumonia-detector
```

### 2. Install Dependencies

```bash
pip install streamlit tensorflow scikit-learn opencv-python pillow \
            numpy matplotlib reportlab joblib
```

Or using a requirements file if provided:

```bash
pip install -r requirements.txt
```

### 3. Verify Model Files

Ensure the following files are present in the `models/` directory:

```
models/
├── densenet121_best_model.keras
├── Gradient_Boost.pkl
└── gradcam.py
```

> The CNN model was trained by **Labani Nabila Nour El Houda** (DL Engineer).  
> The Gradient Boosting model and data preprocessing pipeline were developed by **Fatima Kassouar** (ML Engineer) and **Bouhmidi Amina Maroua** (Data Engineer).  
> The explanation modules (`whyvitals.py`, `whyxray.py`) are located in `apppy/` and require no additional dependencies.

### 4. Run the Application

```bash
streamlit run apppy/app.py
```

The application will open in your browser at `http://localhost:8501`.

---

## Usage Guide

### Step 1 — Complete Your Doctor Profile
On first launch, you will see a profile warning. Click **Profile** and fill in your credentials (name, hospital, region, specialization). Press **Save Changes**. This creates your personal data directory.

### Step 2 — Register a Patient
Click **Add Patient** from the home screen. Fill in the patient's personal information, contact details, and clinical background. Upload optional PDFs for medical history, family illnesses, or clinical notes. Press **Save**. A patient folder is created with a unique ID, along with a CSV and PDF record.

### Step 3 — Run a Diagnostic
Click **Diagnostic** from the home screen. Select a patient.

- **Section A (Vital Signs):** Enter the 12 vital sign fields and press **Analyze Vital Signs**. The Gradient Boosting model returns a prediction and confidence score. `whyvitals.py` then generates a feature-by-feature clinical explanation with active interaction flags and a normalized risk score (0–100).
- **Section B (X-Ray):** Upload a chest X-ray image and press **Start AI Diagnostic Analysis**. DenseNet121 classifies the image and generates Grad-CAM visualizations. `whyxray.py` provides the corresponding radiology explanation with cited signs and clinical notes.
- **Section C (Save Results):** Press **Save All Diagnostic Results** to write the full diagnostic report (CSV + PDF + images) to the patient folder.

### Step 4 — Create a Treatment Plan
Click **Treatment** from the home screen. Select hospitalization status:
- **Hospitalized:** Fill in nursing instructions, surveillance schedule, medications, and discharge conditions.
- **Home Treatment:** Fill in prescribed medications, next appointment, emergency warning signs, and handover notes.

Press **Finalize Treatment Plan** to save.

---

## Team

| Name | Role | Institution |
|---|---|---|
| Dr. Abderrahmane Khiat | Supervisor | University of Saida, Algeria |
| Kassouar Fatima | Project Manager & ML Engineer | University of Saida, Algeria |
| Miloudi Maroua Amira | Fullstack Developer & Business Model | University of Saida, Algeria |
| Bouhmidi Amina Maroua | Data Engineer | University of Saida, Algeria |
| Labani Nabila Nour El Houda | Deep Learning Engineer | University of Saida, Algeria |
| Dr. Aimer Mohammed Djamel Eddine | Medical Advisor | CHU Saida, Algeria |

---

## Acknowledgements

Special thanks to the individual contributors whose work forms the core AI backbone of this system:

- **Labani Nabila Nour El Houda** — CNN model training, DenseNet121 fine-tuning, and Grad-CAM implementation  
  GitHub: [@labaninabila193-code](https://github.com/labaninabila193-code)

- **Bouhmidi Amina Maroua** — Dataset preprocessing and data engineering pipeline  
  GitHub: [@AminaMar](https://github.com/AminaMar)

- **Kassouar Fatima** — Data preprocessing (CSV pipeline), Gradient Boosting algorithm development, and `whyvitals.py` explanation engine  
  GitHub: [@fatimakassouar](https://github.com/fatimakassouar)

---

## Technical Notes & Known Issues

- The `whyvitals.py` risk score (0–100) is an **internal interpretability metric** and is not directly equivalent to the Gradient Boosting model's confidence percentage. They are complementary outputs.
- The `"uncertain"` key in `whyxray.py` must be triggered programmatically in `Diagnostic.py` when `abs(pred_score - 0.260) < threshold_margin`. It is not returned automatically by the model.
- Both `whyvitals.py` and `whyxray.py` are **stateless pure-Python modules** — they have no Streamlit imports and can be unit-tested independently of the UI.
- The `VitalInput` dataclass in `whyvitals.py` expects `Age` as an integer (1–16) and `Oxygen_saturation` / `Temperature` as floats. Type mismatches will raise `AttributeError` from `getattr`.

---

## License

This project was developed as an academic capstone project at the **University of Saida, Algeria, 2026**. All rights reserved by the project team. Contact the supervisor for usage permissions.

---

*University of Saida, Algeria · 2026 · Powered by DenseNet121, Gradient Boosting & Medically-Referenced AI Explainability*
