# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import traceback
from pathlib import Path
from datetime import datetime

# ---------------------------
# Config
# ---------------------------
APP_TITLE = "🏦 Loan Approval Prediction"
SUBTITLE = "An intelligent ML-powered system to assist loan decisions."
MODEL_PATHS = ["loan_approval_model.pkl", "loan_approval_model.joblib", "model/loan_approval_model.pkl"]
GITHUB_URL = "https://github.com/AdityaJadhav-ds"
LINKEDIN_URL = "https://www.linkedin.com/in/aditya-jadhav-6775702b4"
# Use latest user-uploaded screenshot path from conversation history
SCREENSHOT_PATH = "/mnt/data/2025-11-24T08-26-14.423Z.png"
ALLOWED_TERM_OPTIONS = [12, 36, 60, 120, 180, 240, 300, 360, 480]

# ---------------------------
# Page & CSS (faint blue gradient, accessible)
# ---------------------------
st.set_page_config(page_title=APP_TITLE, page_icon="🏦", layout="wide")

CSS = """
<style>
/* Faint-blue premium background (soft, non-distracting) */
.stApp {
  background: linear-gradient(180deg,#f2f8ff,#ddeaff);
  color: #000000;
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
}

/* Headings */
h1, h2, h3, h4 { color: #06243a !important; font-weight:800; }

/* Card container */
.app-card {
  background: #ffffff;
  padding: 20px;
  border-radius: 12px;
  box-shadow: 0 8px 30px rgba(6,36,58,0.06);
  border: 1px solid rgba(6,36,58,0.04);
}

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#052233,#0b2f42);
  color: #e6f6ff;
  padding: 18px;
}
[data-testid="stSidebar"] a { color: #9be7ff !important; text-decoration: none; }

/* Buttons */
.stButton>button {
  background-color: #0b4670 !important;
  color: #ffffff !important;
  padding: 10px 18px !important;
  border-radius: 10px !important;
  font-weight:700;
  border: none !important;
}
button[kind="secondary"] { background-color: #2c3e50 !important; color: #ffffff !important; }

/* DataFrame header contrast */
.stDataFrame thead th { background-color: #f1f5f9 !important; color:#06243a !important; }

/* Result cards */
.result-card { padding:16px; border-radius:10px; text-align:center; box-shadow: 0 6px 20px rgba(2,6,23,0.04); }
.success-card { background: linear-gradient(90deg,#e6ffef,#f0f9ff); border-left:6px solid #10b981; }
.fail-card { background: linear-gradient(90deg,#fff1f0,#fdf2f8); border-left:6px solid #ef4444; }

.muted { color:#475569; }
.small { font-size:13px; color:#475569; }

/* INPUT & LABEL STYLING - enforce light backgrounds and black text for perfect readability */
.stApp input, .stApp select, .stApp textarea,
.stApp [role="combobox"], .stApp [role="listbox"], .stApp [data-baseweb="select"] {
  background-color: #ffffff !important;
  color: #000000 !important;
  border-radius: 8px !important;
  padding: 8px !important;
  border: 1px solid rgba(6,36,58,0.08) !important;
  -webkit-text-fill-color: #000000 !important;
}

/* Number inputs */
input[type="number"] { background-color:#ffffff !important; color:#000000 !important; }

/* Dropdown options readable */
select option, option { color: #000000 !important; background-color: #fff !important; }

/* Force labels and small helper text to black */
label, .stText, .stMarkdown, .stMetricLabel {
  color: #000000 !important;
  -webkit-text-fill-color: #000000 !important;
}

/* Prevent blue selection chips: neutral selection */
::selection { background: rgba(11,70,112,0.12); color: #000000; }
::-moz-selection { background: rgba(11,70,112,0.12); color: #000000; }

/* Remove overly dark widget box backgrounds forced by some themes */
div[data-testid^="stDropdown"] > div, div[class*="stSelectbox"], div[class*="stNumberInput"], div[class*="stTextInput"], div[class*="stMultiSelect"] {
  background-color: transparent !important;
  padding: 0 !important;
}

/* small info circle */
.info-circle {
  display:inline-block;
  width:20px;height:20px;border-radius:50%;background:#eef6ff;color:#0b4670;text-align:center;font-weight:700;margin-left:8px;font-size:12px;
}

/* focus outline gentle */
*:focus { outline: 2px solid rgba(11,70,112,0.12) !important; box-shadow: none !important; }

@media (max-width:900px) { h1{ font-size:1.5rem; } .app-card{ padding:14px; } }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------
# Utilities & model loader
# ---------------------------
def _try_load_model_from_path(pth: Path):
    try:
        return joblib.load(pth)
    except Exception:
        with open(pth, "rb") as f:
            return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_model(paths=MODEL_PATHS):
    last_err = None
    for p in paths:
        pth = Path(p)
        if not pth.exists():
            continue
        try:
            obj = _try_load_model_from_path(pth)
            result = {"model": None, "preprocessor": None, "scaler": None, "encoders": None, "metadata": {}}
            if isinstance(obj, dict):
                result["model"] = obj.get("model") or obj.get("estimator")
                result["preprocessor"] = obj.get("preprocessor")
                result["scaler"] = obj.get("scaler")
                result["encoders"] = obj.get("encoders") or obj.get("label_encoders")
                result["metadata"] = obj.get("metadata", {})
            elif isinstance(obj, tuple):
                if len(obj) == 3:
                    result["model"], result["scaler"], result["encoders"] = obj
                elif len(obj) == 2:
                    result["model"], result["metadata"]["sklearn_version"] = obj
                else:
                    result["model"] = obj[0]
            else:
                result["model"] = obj
            return result
        except Exception as e:
            last_err = e
            continue
    raise FileNotFoundError("No model found in paths {}. Last error: {}".format(paths, last_err))

@st.cache_data(show_spinner=False)
def read_csv_safe(file) -> pd.DataFrame:
    return pd.read_csv(file)

def df_to_download_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode()

def safe_encode_inputs(df: pd.DataFrame, encoders):
    if not encoders:
        return df
    df_copy = df.copy()
    for col, enc in encoders.items():
        if col in df_copy.columns:
            try:
                df_copy[col] = enc.transform(df_copy[col].astype(str))
            except Exception:
                try:
                    df_copy[col] = enc.transform(df_copy[col])
                except Exception:
                    pass
    return df_copy

# ---------------------------
# Load model
# ---------------------------
with st.spinner("Loading model..."):
    try:
        bundle = load_model()
        model = bundle.get("model")
        preprocessor = bundle.get("preprocessor")
        scaler = bundle.get("scaler")
        encoders = bundle.get("encoders")
        metadata = bundle.get("metadata", {})
        sklearn_version = metadata.get("sklearn_version", None)
    except FileNotFoundError:
        st.sidebar.error("⚠️ Model not found. Upload model (.pkl/.joblib) to the app folder.")
        model = preprocessor = scaler = encoders = None
        metadata = {}
        sklearn_version = None
    except Exception as e:
        st.sidebar.error("❌ Failed to load model.")
        st.sidebar.exception(e)
        model = preprocessor = scaler = encoders = None
        metadata = {}
        sklearn_version = None

# ---------------------------
# Header & subtitle
# ---------------------------
st.markdown("<div style='text-align:center; margin-top:8px'><h1>{}</h1></div>".format(APP_TITLE), unsafe_allow_html=True)
st.markdown("<div style='text-align:center; margin-bottom:20px'><h4 style='margin-top:6px; color:#0b4670'>{}</h4></div>".format(SUBTITLE), unsafe_allow_html=True)

# ---------------------------
# Sidebar (icon links + developer panel)
# ---------------------------
with st.sidebar:
    st.markdown("<h3 style='color:#e6f6ff'>Controls</h3>", unsafe_allow_html=True)
    input_mode = st.radio("Input mode", ("Single", "Batch"), index=0)
    st.markdown("---")
    # inline SVG icons for GitHub & LinkedIn (soft blue tint)
    github_svg = '<svg height="20" viewBox="0 0 16 16" width="20" xmlns="http://www.w3.org/2000/svg"><path fill="#9be7ff" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.54 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.28.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.19 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'
    linkedin_svg = '<svg height="20" viewBox="0 0 34 34" width="20" xmlns="http://www.w3.org/2000/svg"><path fill="#9be7ff" d="M34,3.5 C34,1.6 32.4,0 30.5,0 L3.5,0 C1.6,0 0,1.6 0,3.5 L0,30.5 C0,32.4 1.6,34 3.5,34 L30.5,34 C32.4,34 34,32.4 34,30.5 L34,3.5 Z M10.8,28.3 L6,28.3 L6,12.8 L10.8,12.8 L10.8,28.3 Z M8.4,10.6 C6.9,10.6 5.7,9.4 5.7,7.9 C5.7,6.4 6.9,5.1 8.4,5.1 C9.9,5.1 11.1,6.4 11.1,7.9 C11.1,9.4 9.9,10.6 8.4,10.6 Z M29.3,28.3 L24.5,28.3 L24.5,20.6 C24.5,18.6 23.2,17.6 21.7,17.6 C20.3,17.6 19.4,18.8 19.1,19.6 C19.1,19.6 19.1,28.3 19.1,28.3 L14.3,28.3 L14.3,12.8 L18.9,12.8 L18.9,14.6 C19.7,13.5 21.4,11.8 24.6,11.8 C29,11.8 29.3,15.2 29.3,19.6 L29.3,28.3 Z"/></svg>'
    st.markdown(
        "<div style='display:flex; gap:12px; align-items:center; margin-bottom:12px'><a href='{}' target='_blank' title='GitHub'>{}</a><a href='{}' target='_blank' title='LinkedIn'>{}</a></div>".format(
            GITHUB_URL, github_svg, LINKEDIN_URL, linkedin_svg
        ),
        unsafe_allow_html=True,
    )

    with st.expander("Developer: model & screenshot", expanded=False):
        if model is not None:
            st.write("Model class:", getattr(model, "__class__", type(model)).__name__)
            if sklearn_version:
                st.write("scikit-learn:", sklearn_version)
            st.json(metadata)
        else:
            st.write("Model not loaded.")
        # show screenshot for visual debugging
        try:
            st.markdown("**Uploaded screenshot (for UI debugging):**")
            st.image(SCREENSHOT_PATH, use_column_width=True)
        except Exception:
            st.write("No screenshot available in this environment.")

    st.markdown("---")
    st.caption("Tip: For batch predictions upload a CSV with the exact column names used in training (see sample).")
    if st.button("Download sample CSV"):
        sample = pd.DataFrame(
            [
                {
                    "Gender": "Male",
                    "Married": "Yes",
                    "Dependents": "0",
                    "Education": "Graduate",
                    "Self_Employed": "No",
                    "ApplicantIncome": 5000,
                    "CoapplicantIncome": 0,
                    "LoanAmount": 120,
                    "Loan_Amount_Term": 360,
                    "Credit_History": 1,
                    "Property_Area": "Urban",
                }
            ]
        )
        st.download_button("Download sample.csv", data=df_to_download_bytes(sample), file_name="loan_sample_input.csv", mime="text/csv")

# ---------------------------
# Main: input card
# ---------------------------
st.markdown("<div class='app-card'>", unsafe_allow_html=True)
st.markdown("## Applicant details")
st.markdown("Fill applicant information below. Use the form and click Predict.")

with st.form(key="single_form", clear_on_submit=False):
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0, help="Applicant gender")
        married = st.selectbox("Married", ["Yes", "No"], index=0, help="Marital status")
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=0, help="Number of dependents")
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0, help="Education level")
        self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=0, help="Is the applicant self-employed?")
    with c2:
        applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=500, help="Monthly applicant income")
        coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=500, help="Monthly coapplicant income")
        loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=120, step=1, help="Requested loan amount (in thousands or same units as trained model)")
        loan_amount_term = st.selectbox("Loan Amount Term (months)", ALLOWED_TERM_OPTIONS, index=7, help="Repayment term in months")
        credit_history = st.selectbox("Credit History (1 = good)", [1, 0], index=0, help="1 indicates good credit history")
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=0, help="Property area type")

    submitted = st.form_submit_button("🔮 Predict Loan Approval")
st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Batch upload handling
# ---------------------------
input_df = None
if input_mode == "Batch":
    uploaded = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
    if uploaded is not None:
        try:
            input_df = read_csv_safe(uploaded)
            st.success("Loaded {} rows".format(len(input_df)))
            if st.checkbox("Show uploaded data"):
                st.dataframe(input_df)
        except Exception as e:
            st.error("Failed to read uploaded CSV.")
            st.exception(e)
            st.stop()
else:
    input_df = pd.DataFrame(
        [
            {
                "Gender": gender,
                "Married": married,
                "Dependents": dependents,
                "Education": education,
                "Self_Employed": self_employed,
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": loan_amount,
                "Loan_Amount_Term": loan_amount_term,
                "Credit_History": credit_history,
                "Property_Area": property_area,
            }
        ]
    )
    if st.checkbox("Show input data"):
        st.dataframe(input_df)

st.write("---")

# ---------------------------
# Prepare & predict helpers
# ---------------------------
def prepare_for_model(df: pd.DataFrame):
    X = df.copy()
    X = safe_encode_inputs(X, encoders)
    if scaler is not None:
        try:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                X_num = pd.DataFrame(scaler.transform(X[numeric_cols]), columns=numeric_cols, index=X.index)
                X.update(X_num)
        except Exception:
            pass
    if preprocessor is not None:
        try:
            X_trans = preprocessor.transform(X)
            return X_trans, True
        except Exception:
            pass
    return X, False

def build_results_df(input_df, preds, probs=None):
    results = input_df.copy().reset_index(drop=True)
    results["Loan_Approved"] = np.where(np.asarray(preds).astype(int) == 1, "Approved", "Rejected")
    if probs is not None:
        results["Approval_Confidence"] = (probs * 100).round(2)
    return results

# ---------------------------
# Run predictions
# ---------------------------
if (input_mode == "Batch" and 'uploaded' in locals() and uploaded is not None and st.button("Run Batch Predictions")) or (input_mode == "Single" and submitted):
    with st.spinner("Predicting..."):
        try:
            if model is None:
                st.error("Model not available. Please upload model file to app folder (see sidebar).")
            else:
                X_ready, transformed = prepare_for_model(input_df)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_ready)
                    preds = model.predict(X_ready)
                    if probs.shape[1] == 2:
                        positive_probs = probs[:, 1]
                    else:
                        positive_probs = probs.max(axis=1)
                else:
                    preds = model.predict(X_ready)
                    positive_probs = None

                results = build_results_df(input_df, preds, positive_probs)

                if len(results) == 1:
                    row = results.loc[0]
                    status = row["Loan_Approved"]
                    confidence = row.get("Approval_Confidence", None)
                    if status == "Approved":
                        st.markdown("<div class='result-card success-card'><h2>✅ Loan Approved</h2><p class='muted'>Confidence: <b>{}</b></p></div>".format(str(confidence) + "%" if confidence is not None else "—"), unsafe_allow_html=True)
                    else:
                        st.markdown("<div class='result-card fail-card'><h2>❌ Loan Rejected</h2><p class='muted'>Confidence: <b>{}</b></p></div>".format(str(confidence) + "%" if confidence is not None else "—"), unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1, 1, 1])
                    with col1:
                        st.metric("Prediction", status)
                    with col2:
                        st.metric("Confidence", "{}".format(str(confidence) + "%" if confidence is not None else "—"))
                    with col3:
                        st.metric("Model", getattr(model, "__class__", type(model)).__name__)
                else:
                    approved_count = (results["Loan_Approved"] == "Approved").sum()
                    total = len(results)
                    st.markdown("<div class='result-card'><h3>Batch prediction completed</h3><p class='muted'>Approved: <b>{}</b> / {}</p></div>".format(approved_count, total), unsafe_allow_html=True)

                st.markdown("### Detailed results")
                st.dataframe(results)
                csv_bytes = df_to_download_bytes(results)
                st.download_button("⬇️ Download predictions (CSV)", data=csv_bytes, file_name="loan_predictions_{}.csv".format(datetime.utcnow().strftime("%Y%m%d_%H%M%S")), mime="text/csv")

                # optional feature importances (approx)
                try:
                    fi = None
                    if hasattr(model, "feature_importances_"):
                        fi = np.asarray(model.feature_importances_)
                    elif hasattr(model, "coef_"):
                        fi = np.abs(np.asarray(model.coef_)).ravel()

                    if fi is not None:
                        try:
                            if transformed and preprocessor is not None:
                                feature_names = preprocessor.get_feature_names_out()
                            else:
                                feature_names = input_df.columns
                        except Exception:
                            feature_names = input_df.columns

                        if len(feature_names) == len(fi):
                            fi_df = pd.DataFrame({"feature": feature_names, "importance": fi})
                            fi_df = fi_df.sort_values("importance", ascending=False).head(10).reset_index(drop=True)
                            st.markdown("---")
                            st.markdown("#### Top feature importances (approx)")
                            st.table(fi_df)
                except Exception:
                    pass

        except Exception as e:
            st.error("Prediction failed. See developer panel for details.")
            st.sidebar.exception(traceback.format_exc())

# ---------------------------
# Footer
# ---------------------------
st.write("---")
st.markdown(
    "<div style='text-align:center; color:#475569; font-size:13px'>Built with ❤️ by <b>Aditya Jadhav</b> · <a href='{}' target='_blank'>GitHub</a> · <a href='{}' target='_blank'>LinkedIn</a><br><span style='font-size:12px'>v1.3 • © {}</span></div>".format(
        GITHUB_URL, LINKEDIN_URL, datetime.utcnow().year
    ),
    unsafe_allow_html=True,
)
