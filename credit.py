import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
from pathlib import Path

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & CUSTOM FINTECH STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Aegis Quant | Credit Risk Terminal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    /* Global Dark Obsidian Theme */
    .stApp {
        background: radial-gradient(circle at 10% 10%, rgba(99, 102, 241, 0.12), transparent 35%),
                    radial-gradient(circle at 90% 90%, rgba(16, 185, 129, 0.10), transparent 35%),
                    #07090e;
        color: #f8fafc;
        font-family: 'Inter', sans-serif;
    }

    /* Top Header Banner */
    .terminal-header {
        background: rgba(15, 20, 32, 0.82);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 20px 28px;
        margin-bottom: 24px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 16px;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
    }
    .brand-title {
        font-size: 1.5rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        margin: 0;
        color: #f8fafc;
    }
    .brand-title span {
        color: #38bdf8;
        font-weight: 400;
    }
    .brand-sub {
        color: #94a3b8;
        font-size: 0.85rem;
        margin: 2px 0 0 0;
    }
    .pill-cluster {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .Sys-pill {
        background: rgba(9, 13, 22, 0.9);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        color: #cbd5e1;
        font-family: 'JetBrains Mono', monospace;
    }
    .Sys-pill strong {
        color: #38bdf8;
    }

    /* Glass Panels & Cards */
    .glass-card {
        background: rgba(15, 20, 32, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.35);
    }
    .section-tag {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        font-weight: 600;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.12);
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid rgba(56, 189, 248, 0.25);
        margin-right: 8px;
    }
    .section-heading {
        font-size: 0.95rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #cbd5e1;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
    }

    /* Diagnostic Metric Cards */
    .diag-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-top: 12px;
        margin-bottom: 18px;
    }
    .diag-box {
        background: rgba(9, 13, 22, 0.85);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 12px;
        padding: 14px 16px;
    }
    .diag-label {
        font-size: 0.74rem;
        color: #94a3b8;
        display: block;
    }
    .diag-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 4px 0 2px 0;
        display: block;
    }
    .diag-sub {
        font-size: 0.7rem;
        color: #64748b;
    }

    /* Verdict Banner */
    .verdict-banner {
        padding: 14px 20px;
        border-radius: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        margin-bottom: 16px;
    }
    .verdict-low {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.4);
        color: #10b981;
    }
    .verdict-high {
        background: rgba(239, 68, 68, 0.14);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #ef4444;
    }

    /* Signal List */
    .signal-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        font-size: 0.82rem;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .dot-pos { width: 8px; height: 8px; border-radius: 50%; background: #10b981; box-shadow: 0 0 6px #10b981; flex-shrink: 0; }
    .dot-neg { width: 8px; height: 8px; border-radius: 50%; background: #ef4444; box-shadow: 0 0 6px #ef4444; flex-shrink: 0; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. SMART MODEL & THRESHOLD LOADER
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

def find_file(candidates):
    for name in candidates:
        p = BASE_DIR / name
        if p.exists():
            return p
    return None

@st.cache_resource
def load_artifacts():
    # Expanded list of potential model names based on standard formatting
    model_candidates = [
        "credit_risk_model.pkl",
        "Credit_Risk_Assesment.pkl",
        "Credit_Risk_Assesment (1).pkl",
        "model.pkl",
        "xgboost_model.pkl"
    ]
    
    thresh_candidates = [
        "best_threshold.pkl",
        "Best_Threshold.pkl",
        "Best_Threshold (1).pkl",
        "threshold.pkl"
    ]

    model_path = find_file(model_candidates)
    thresh_path = find_file(thresh_candidates)

    if model_path is None:
        files_in_dir = [f.name for f in BASE_DIR.iterdir() if f.is_file()]
        raise FileNotFoundError(
            f"Could not find the model .pkl file! \n\n"
            f"Please rename your model file to 'credit_risk_model.pkl' and ensure it is in this folder: {BASE_DIR}\n\n"
            f"Files currently in this folder: {files_in_dir}"
        )

    model = joblib.load(model_path)
    threshold = float(joblib.load(thresh_path)) if thresh_path else 0.267986
    return model, threshold, model_path.name

try:
    ml_model, opt_threshold, loaded_model_name = load_artifacts()
except Exception as e:
    st.error(f"⚠️ **Model Loading Error:** {e}")
    st.stop()

# -----------------------------------------------------------------------------
# 3. SESSION STATE & PRESET PROFILES
# -----------------------------------------------------------------------------
PRESETS = {
    "prime": {
        "person_age": 32,
        "person_emp_length": 8.0,
        "person_income": 115000.0,
        "person_home_ownership": "MORTGAGE",
        "loan_amnt": 10000.0,
        "loan_int_rate": 7.49,
        "loan_intent": "VENTURE",
        "loan_grade": "A",
        "cb_person_cred_hist_length": 10,
        "cb_person_default_on_file": "N",
    },
    "borderline": {
        "person_age": 24,
        "person_emp_length": 2.0,
        "person_income": 48000.0,
        "person_home_ownership": "RENT",
        "loan_amnt": 12000.0,
        "loan_int_rate": 12.87,
        "loan_intent": "DEBTCONSOLIDATION",
        "loan_grade": "C",
        "cb_person_cred_hist_length": 3,
        "cb_person_default_on_file": "N",
    },
    "subprime": {
        "person_age": 22,
        "person_emp_length": 1.0,
        "person_income": 28000.0,
        "person_home_ownership": "RENT",
        "loan_amnt": 15000.0,
        "loan_int_rate": 17.50,
        "loan_intent": "PERSONAL",
        "loan_grade": "E",
        "cb_person_cred_hist_length": 2,
        "cb_person_default_on_file": "Y",
    }
}

for k, v in PRESETS["prime"].items():
    if k not in st.session_state:
        st.session_state[k] = v

def apply_preset(preset_key):
    for k, v in PRESETS[preset_key].items():
        st.session_state[k] = v

# -----------------------------------------------------------------------------
# 4. TOP NAVIGATION BAR
# -----------------------------------------------------------------------------
st.markdown(f"""
<div class="terminal-header">
    <div>
        <h1 class="brand-title">🛡️ AEGIS<span>QUANT</span></h1>
        <p class="brand-sub">Institutional Credit Risk & Default Underwriting Engine</p>
    </div>
    <div class="pill-cluster">
        <div class="Sys-pill">🟢 Model: <strong>Calibrated XGBoost</strong></div>
        <div class="Sys-pill">ROC-AUC: <strong>0.945</strong></div>
        <div class="Sys-pill">Decision Cutoff: <strong>{opt_threshold*100:.2f}%</strong></div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. MAIN TWO-COLUMN WORKSPACE
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([1.15, 0.85], gap="large")

with left_col:
    st.markdown('<div class="section-heading"><span class="section-tag">PRESETS</span> Benchmark Borrower Profiles</div>', unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    with p1:
        st.button("🟢 Load Prime Tier", on_click=apply_preset, args=("prime",), use_container_width=True)
    with p2:
        st.button("🟡 Load Borderline", on_click=apply_preset, args=("borderline",), use_container_width=True)
    with p3:
        st.button("🔴 Load High Risk", on_click=apply_preset, args=("subprime",), use_container_width=True)

    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 16px 0;'>", unsafe_allow_html=True)

    # Section 01: Demographics & Income
    st.markdown('<div class="section-heading"><span class="section-tag">01</span> Applicant Demographics & Income</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        person_age = st.number_input("Applicant Age (Years)", min_value=18, max_value=100, step=1, key="person_age")
        person_income = st.number_input("Annual Gross Income ($ USD)", min_value=4000.0, max_value=10000000.0, step=1000.0, key="person_income")
    with c2:
        person_emp_length = st.number_input("Employment Tenure (Years)", min_value=0.0, max_value=60.0, step=0.5, key="person_emp_length")
        person_home_ownership = st.selectbox(
            "Home Ownership Status",
            options=["MORTGAGE", "RENT", "OWN", "OTHER"],
            key="person_home_ownership"
        )

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

    # Section 02: Loan Facility Terms
    st.markdown('<div class="section-heading"><span class="section-tag">02</span> Facility & Underwriting Terms</div>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    with c3:
        loan_amnt = st.number_input("Requested Principal ($ USD)", min_value=500.0, max_value=100000.0, step=500.0, key="loan_amnt")
        loan_intent = st.selectbox(
            "Loan Purpose / Intent",
            options=["VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT", "EDUCATION", "MEDICAL", "PERSONAL"],
            key="loan_intent"
        )
    with c4:
        loan_int_rate = st.number_input("Interest Rate (APR %)", min_value=1.0, max_value=40.0, step=0.1, key="loan_int_rate")
        auto_lti = round(min(1.0, loan_amnt / person_income), 2) if person_income > 0 else 0.0
        loan_percent_income = st.number_input(
            "Loan-to-Income Ratio (Auto-Calculated)",
            min_value=0.0,
            max_value=1.0,
            value=auto_lti,
            step=0.01,
            disabled=False
        )

    loan_grade = st.select_slider(
        "Assigned Credit Grade (A = Prime Tier, G = Deep Subprime)",
        options=["A", "B", "C", "D", "E", "F", "G"],
        key="loan_grade"
    )

    st.markdown("<div style='height: 12px'></div>", unsafe_allow_html=True)

    # Section 03: Bureau Telemetry
    st.markdown('<div class="section-heading"><span class="section-tag">03</span> Credit Bureau Telemetry</div>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)
    with c5:
        cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", min_value=0, max_value=50, step=1, key="cb_person_cred_hist_length")
    with c6:
        cb_person_default_on_file = st.radio(
            "Prior Default on Bureau File",
            options=["N", "Y"],
            format_func=lambda x: "Clean Record (N)" if x == "N" else "Prior Default Recorded (Y)",
            horizontal=True,
            key="cb_person_default_on_file"
        )

# -----------------------------------------------------------------------------
# 6. REAL-TIME PREDICTION & TELEMETRY DASHBOARD
# -----------------------------------------------------------------------------
input_df = pd.DataFrame([{
    "person_age": int(person_age),
    "person_income": float(person_income),
    "person_home_ownership": str(person_home_ownership),
    "person_emp_length": float(person_emp_length),
    "loan_intent": str(loan_intent),
    "loan_grade": str(loan_grade),
    "loan_amnt": float(loan_amnt),
    "loan_int_rate": float(loan_int_rate),
    "loan_percent_income": float(loan_percent_income),
    "cb_person_default_on_file": str(cb_person_default_on_file),
    "cb_person_cred_hist_length": int(cb_person_cred_hist_length)
}])

prob = float(ml_model.predict_proba(input_df)[:, 1][0])
prediction = int(prob >= opt_threshold)
is_high_risk = prediction == 1

prob_pct = prob * 100
thresh_pct = opt_threshold * 100
spread_pct = prob_pct - thresh_pct
annual_interest = loan_amnt * (loan_int_rate / 100.0)

with right_col:
    # Verdict Banner
    verdict_class = "verdict-high" if is_high_risk else "verdict-low"
    verdict_title = "🔴 HIGH RISK // DECLINE FACILITY" if is_high_risk else "🟢 LOW RISK // APPROVE FACILITY"
    st.markdown(f"""
    <div class="verdict-banner {verdict_class}">
        <span>{verdict_title}</span>
        <span>CLASS: {prediction}</span>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Plotly Speedometer Gauge
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prob_pct,
        number={'suffix': "%", 'valueformat': ".2f", 'font': {'size': 42, 'color': '#f8fafc', 'family': 'JetBrains Mono'}},
        delta={
            'reference': thresh_pct,
            'position': "bottom",
            'valueformat': ".2f",
            'increasing': {'color': "#ef4444"},
            'decreasing': {'color': "#10b981"}
        },
        title={'text': f"DEFAULT PROBABILITY (vs {thresh_pct:.2f}% Cutoff)", 'font': {'size': 13, 'color': '#94a3b8'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#64748b"},
            'bar': {'color': "#ef4444" if is_high_risk else "#10b981", 'thickness': 0.32},
            'bgcolor': "rgba(15, 20, 32, 0.9)",
            'borderwidth': 1,
            'bordercolor': "rgba(255, 255, 255, 0.08)",
            'steps': [
                {'range': [0, thresh_pct], 'color': "rgba(16, 185, 129, 0.14)"},
                {'range': [thresh_pct, min(100, thresh_pct * 1.75)], 'color': "rgba(245, 158, 11, 0.14)"},
                {'range': [min(100, thresh_pct * 1.75), 100], 'color': "rgba(239, 68, 68, 0.16)"}
            ],
            'threshold': {
                'line': {'color': "#38bdf8", 'width': 4},
                'thickness': 0.82,
                'value': thresh_pct
            }
        }
    ))
    gauge_fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=24, r=24, t=45, b=15),
        height=265
    )
    st.plotly_chart(gauge_fig, use_container_width=True)

    # 4-Card Quantitative Diagnostics Grid
    spread_color = "#ef4444" if spread_pct >= 0 else "#10b981"
    lti_color = "#ef4444" if loan_percent_income > 0.35 else ("#f59e0b" if loan_percent_income > 0.20 else "#10b981")
    lti_status = "Elevated Debt Burden" if loan_percent_income > 0.35 else ("Moderate Leverage" if loan_percent_income > 0.20 else "Healthy Leverage")

    st.markdown(f"""
    <div class="diag-grid">
        <div class="diag-box">
            <span class="diag-label">Underwriting Decision</span>
            <span class="diag-value" style="color: {'#ef4444' if is_high_risk else '#10b981'};">{"High Risk" if is_high_risk else "Low Risk"}</span>
            <span class="diag-sub">F1-Optimal Cutoff: {thresh_pct:.2f}%</span>
        </div>
        <div class="diag-box">
            <span class="diag-label">Threshold Spread</span>
            <span class="diag-value" style="color: {spread_color};">{spread_pct:+.2f}%</span>
            <span class="diag-sub">{"Above Risk Limit" if spread_pct >= 0 else "Safety Buffer Margin"}</span>
        </div>
        <div class="diag-box">
            <span class="diag-label">Loan-to-Income (LTI)</span>
            <span class="diag-value" style="color: {lti_color};">{loan_percent_income*100:.1f}%</span>
            <span class="diag-sub">{lti_status}</span>
        </div>
        <div class="diag-box">
            <span class="diag-label">Est. Annual Interest</span>
            <span class="diag-value">${annual_interest:,.0f}</span>
            <span class="diag-sub">At {loan_int_rate:.2f}% Coupon APR</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Underwriting Signal Breakdown
    st.markdown('<div class="section-heading"><span class="section-tag">SIGNALS</span> Underwriting Risk Factors</div>', unsafe_allow_html=True)
    signals = []
    if loan_percent_income >= 0.30:
        signals.append(("neg", f"High Loan-to-Income ratio ({loan_percent_income*100:.0f}%) significantly elevates default probability."))
    else:
        signals.append(("pos", f"Conservative Loan-to-Income ratio ({loan_percent_income*100:.0f}%) supports debt service coverage."))

    if loan_grade in ["D", "E", "F", "G"]:
        signals.append(("neg", f"Subprime Credit Grade ({loan_grade}) indicates elevated historical delinquency risk."))
    else:
        signals.append(("pos", f"Prime/Near-Prime Credit Grade ({loan_grade}) signals strong borrower creditworthiness."))

    if cb_person_default_on_file == "Y":
        signals.append(("neg", "Prior default on credit bureau file heavily penalizes underwriting score."))
    else:
        signals.append(("pos", "Clean bureau record with no prior historical defaults on file."))

    if loan_int_rate >= 14.0:
        signals.append(("neg", f"High APR coupon ({loan_int_rate:.2f}%) increases annual interest carry burden."))

    for sig_type, text in signals:
        dot_cls = "dot-pos" if sig_type == "pos" else "dot-neg"
        st.markdown(f'<div class="signal-item"><span class="{dot_cls}"></span><span>{text}</span></div>', unsafe_allow_html=True)