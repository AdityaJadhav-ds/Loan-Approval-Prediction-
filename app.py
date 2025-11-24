# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import traceback
from pathlib import Path
from datetime import datetime

# ---------------------------
# Configuration & constants
# ---------------------------
APP_TITLE = "🏦 Loan Approval Prediction"
MODEL_PATHS = ["loan_approval_model.pkl", "loan_approval_model.joblib", "model/loan_approval_model.pkl"]
GITHUB_URL = "https://github.com/AdityaJadhav-ds"
LINKEDIN_URL = "https://www.linkedin.com/in/aditya-jadhav-6775702b4"
ALLOWED_TERM_OPTIONS = [12, 36, 60, 120, 180, 240, 300, 360, 480]
# Path to uploaded screenshot (for developer debugging)
SCREENSHOT_PATH = "/mnt/data/2025-11-24T08-17-23.628Z.png"

# ---------------------------
# Page setup & CSS (fix input visibility)
# ---------------------------
st.set_page_config(page_title="Loan Approval - Aditya Jadhav", page_icon="🏦", layout="wide")

st.markdown(
    """
    <style>
    /* Base app colors & fonts */
    .stApp {
        background: linear-gradient(180deg,#fbfdff,#eef4f8);
        color: #052038;
        font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    /* Headings */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #052038 !important;
    }

    /* Card containers */
    .card {
        background: #ffffff;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(2,6,23,0.06);
    }

    /* Sidebar contrast */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#061421,#082534);
        color: #e6f6ff;
        padding: 20px;
    }
    [data-testid="stSidebar"] a { color: #7dd3fc !important; }

    /* Result cards */
    .result-card { 
        padding:16px; 
        border-radius:10px;
        box-shadow: 0 6px 20px rgba(2,6,23,0.04);
        text-align:center;
    }
    .success-card { background: linear-gradient(90deg,#ecfdf5,#f0f9ff); border-left:6px solid #10b981; }
    .fail-card { background: linear-gradient(90deg,#fff1f0,#fdf2f8); border-left:6px solid #ef4444; }

    .muted { color:#475569; }
    .small { font-size:13px; color:#475569; }
    .dev-footer { text-align:center; margin-top:18px; color:#475569; font-size:13px; }

    /* Improve table header contrast */
    .stDataFrame thead th { background-color: #f1f5f9 !important; color:#052038 !important; }

    /* >>> INPUT WIDGET FIXES: make widget background light and text dark <<<
       These selectors are broad to override streamlit theme / cloud styles.
       They force readable backgrounds and dark text for select, inputs and dropdowns.
    */
    .stApp input, .stApp select, .stApp textarea,
    .stApp [role="combobox"], .stApp [role="listbox"], .stApp [data-baseweb="select"] {
        background-color: #ffffff !important;
        color: #052038 !important;
        border-radius: 8px !important;
        padding: 8px !important;
        border: 1px solid rgba(2,6,23,0.06) !important;
    }

    /* For the numeric inputs that show +/- controls */
    input[type="number"] {
        background-color: #ffffff !important;
        color: #052038 !important;
    }

    /* Streamlit often wraps widgets inside divs with classes - make them readable too */
    div[class*="stSelectbox"], div[class*="stNumberInput"], div[class*="stTextInput"],
    div[class*="stMultiSelect"], div[class*="stSlider"] {
        background-color: transparent !important;
    }

    /* Make buttons stand out */
    .stButton>button {
        background-color: #0b6b9d !important;
        color: #ffffff !important;
        padding: 8px 18px !important;
        border-radius: 10px !important;
        font-weight: 600;
    }
    /* Secondary smaller download buttons */
    button[kind="secondary"] {
        background-color: #334155 !important;
        color: #fff !important;
    }

    /* Reduce overly dark focus outlines that sometimes hide text */
    *:focus {
        outline: 2px solid rgba(11,107,157,0.12) !important;
        box-shadow: none !important;
    }

    @media (max-width: 900px) {
        h1 { font-size: 1.5rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Utilities
# ---------------------------
def _try_load_model_from_path(pth: Path):
    """Helper to attempt load with joblib then pickle."""
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
    raise FileNotFoundError(f"No model found in paths {paths}. Last error: {last_err}")

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
        st.sidebar.error("⚠️ Model not found. Upload `loan_approval_model.pkl` (or joblib) to the app folder.")
        model = None
        preprocessor = None
        scaler = None
        encoders = None
        metadata = {}
    except Exception as e:
        st.sidebar.error("❌ Failed to load model. Check model file.")
        st.sidebar.exception(e)
        model = None
        preprocessor = None
        scaler = None
        encoders = None
        metadata = {}

# ---------------------------
# Header (no noisy model text)
# ---------------------------
st.markdown(f"<div style='text-align:center'><h1 style='margin-bottom:6px'>{APP_TITLE}</h1></div>", unsafe_allow_html=True)
st.write("")  # spacer

# ---------------------------
# Sidebar (controls + dev info)
# ---------------------------
with st.sidebar:
    st.markdown("<div style='padding:6px 0'><h3 style='margin:0; color: #e6f6ff'>Controls</h3></div>", unsafe_allow_html=True)
    input_mode = st.radio("Input mode", ("Single", "Batch"), index=0)
    st.markdown("---")
    st.markdown("**Developer / Links**")
    st.markdown(f"[🔗 GitHub]({GITHUB_URL}) • [🔗 LinkedIn]({LINKEDIN_URL})")
    # Show screenshot in dev expander for debugging
    with st.expander("Model (developer) & Screenshot", expanded=False):
        if model is not None:
            st.write("Model class:", getattr(model, "__class__", type(model)).__name__)
            if sklearn_version:
                st.write("scikit-learn version:", sklearn_version)
            st.json(metadata)
        else:
            st.write("Model not loaded.")
        # show uploaded screenshot for debugging UI problems
        try:
            st.markdown("**Uploaded screenshot (for UI debugging):**")
            st.image(SCREENSHOT_PATH, caption="User-uploaded screenshot", use_column_width=True)
        except Exception:
            st.write("Screenshot not available in this environment.")
    st.markdown("---")
    st.caption("Tip: For batch predictions upload a CSV with the exact column names used in training (see sample).")
    if st.button("Download sample CSV"):
        sample = pd.DataFrame([{
            "Gender":"Male","Married":"Yes","Dependents":"0","Education":"Graduate","Self_Employed":"No",
            "ApplicantIncome":5000,"CoapplicantIncome":0,"LoanAmount":120,"Loan_Amount_Term":360,"Credit_History":1,"Property_Area":"Urban"
        }])
        st.download_button("Download sample.csv", data=df_to_download_bytes(sample), file_name="loan_sample_input.csv", mime="text/csv")

# ---------------------------
# Main input card
# ---------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("## Applicant details")
st.markdown("Fill applicant information below. Use the form and click Predict.")

with st.form(key="single_form", clear_on_submit=False):
    cols = st.columns([1, 1])
    with cols[0]:
        gender = st.selectbox("Gender", ["Male", "Female"], index=0, help="Applicant gender")
        married = st.selectbox("Married", ["Yes", "No"], index=0, help="Marital status")
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"], index=0, help="Number of dependents")
        education = st.selectbox("Education", ["Graduate", "Not Graduate"], index=0, help="Education level")
        self_employed = st.selectbox("Self Employed", ["No", "Yes"], index=0, help="Is the applicant self-employed?")
    with cols[1]:
        applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=5000, step=500, help="Monthly applicant income")
        coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=500, help="Monthly coapplicant income")
        loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=120, step=1, help="Requested loan amount (in thousands or same units as trained model)")
        loan_amount_term = st.selectbox("Loan Amount Term (months)", ALLOWED_TERM_OPTIONS, index=7, help="Repayment term in months")
        credit_history = st.selectbox("Credit History (1 = good)", [1, 0], index=0, help="1 indicates good credit history")
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"], index=0, help="Property area type")

    submitted = st.form_submit_button("🔮 Predict Loan Approval")

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------
# Batch upload
# ---------------------------
input_df = None
if input_mode == "Batch":
    uploaded = st.file_uploader("Upload CSV for batch predictions", type=["csv"])
    if uploaded is not None:
        try:
            input_df = read_csv_safe(uploaded)
            st.success(f"Loaded {len(input_df)} rows")
            if st.checkbox("Show uploaded data"):
                st.dataframe(input_df)
        except Exception as e:
            st.error("Failed to read uploaded CSV.")
            st.exception(e)
            st.stop()
else:
    input_df = pd.DataFrame([{
        "Gender": gender, "Married": married, "Dependents": dependents, "Education": education,
        "Self_Employed": self_employed, "ApplicantIncome": applicant_income, "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount, "Loan_Amount_Term": loan_amount_term, "Credit_History": credit_history,
        "Property_Area": property_area
    }])
    if st.checkbox("Show input data"):
        st.dataframe(input_df)

st.write("---")

# ---------------------------
# Model prep & predict
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

# run predictions
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
                        st.markdown(f"<div class='result-card success-card'><h2>✅ Loan Approved</h2><p class='muted'>Confidence: <b>{confidence if confidence is not None else '—'}%</b></p></div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='result-card fail-card'><h2>❌ Loan Rejected</h2><p class='muted'>Confidence: <b>{confidence if confidence is not None else '—'}%</b></p></div>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([1,1,1])
                    with col1:
                        st.metric("Prediction", status)
                    with col2:
                        st.metric("Confidence", f"{confidence}%" if confidence is not None else "—")
                    with col3:
                        st.metric("Model", getattr(model, "__class__", type(model)).__name__)
                else:
                    approved_count = (results["Loan_Approved"] == "Approved").sum()
                    total = len(results)
                    st.markdown(f"<div class='result-card'><h3>Batch prediction completed</h3><p class='muted'>Approved: <b>{approved_count}</b> / {total}</p></div>", unsafe_allow_html=True)

                st.markdown("### Detailed results")
                st.dataframe(results)
                csv_bytes = df_to_download_bytes(results)
                st.download_button("⬇️ Download predictions (CSV)", data=csv_bytes, file_name=f"loan_predictions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

                # optional approximate feature importances
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
    f"""
<div class='dev-footer'>
  Built with ❤️ by <b>Aditya Jadhav</b> ·
  <a href='{GITHUB_URL}' target='_blank'>GitHub</a> ·
  <a href='{LINKEDIN_URL}' target='_blank'>LinkedIn</a><br>
  <span class='small'>v1.2 • © {datetime.utcnow().year} Loan Approval ML</span>
</div>
""",
    unsafe_allow_html=True,
)
