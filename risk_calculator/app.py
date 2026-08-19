"""
Lupus Nephritis Risk Calculator — Streamlit App
Run: streamlit run risk_calculator/app.py
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib

APP_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(APP_DIR, "models")

YOUDEN_THRESH = {
    "1yr":       0.410,
    "5yr":       0.509,
    "esrd_5yr":  0.482,
    "esrd_10yr": 0.470,
}

st.set_page_config(
    page_title="Risk Calculator",
    page_icon="⚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  /* ── NHS Design System ── */
  html, body, [class*="css"] {
    font-family: Arial, Helvetica, sans-serif;
    color: #212b32;
    background-color: #f0f4f5;
  }

  /* Remove Streamlit default padding at top */
  .block-container { padding-top: 0 !important; }
  header[data-testid="stHeader"] { background: transparent; }

  /* ── NHS Blue header bar ── */
  .nhs-header {
    background-color: #005EB8;
    padding: 16px 24px 14px;
    margin: -1rem -1rem 0 -1rem;
  }
  .nhs-header-title {
    color: #ffffff;
    font-size: 1.4rem;
    font-weight: 700;
    margin: 0 0 2px 0;
    letter-spacing: 0;
  }
  .nhs-header-sub {
    color: #d8e8f7;
    font-size: 0.85rem;
    font-weight: 400;
    margin: 0;
  }

  /* ── White content card ── */
  .nhs-card {
    background: #ffffff;
    padding: 24px 28px;
    margin-top: 20px;
    border-top: 4px solid #005EB8;
  }

  /* ── Intro text ── */
  .nhs-intro {
    font-size: 1rem;
    color: #212b32;
    line-height: 1.6;
    margin: 0 0 20px 0;
  }

  /* ── Section headings ── */
  .nhs-section {
    font-size: 1rem;
    font-weight: 700;
    color: #212b32;
    margin: 24px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid #d8dde0;
  }

  /* ── Result panel ── */
  .nhs-result-low  { padding: 16px 20px; background: #ffffff; margin-bottom: 8px; }
  .nhs-result-mod  { padding: 16px 20px; background: #ffffff; margin-bottom: 8px; }
  .nhs-result-high { padding: 16px 20px; background: #ffffff; margin-bottom: 8px; }

  /* ── Clinical action banner ── */
  .clinical-low  { background:#f0faf4; padding:10px 14px; font-size:0.85rem; color:#212b32; margin-bottom:16px; border-radius:4px; }
  .clinical-mod  { background:#fdf6e3; padding:10px 14px; font-size:0.85rem; color:#212b32; margin-bottom:16px; border-radius:4px; }
  .clinical-high { background:#fdf0f0; padding:10px 14px; font-size:0.85rem; color:#212b32; margin-bottom:16px; border-radius:4px; }

  .nhs-result-pct {
    font-size: 3rem; font-weight: 700; line-height: 1; margin: 0;
  }
  .nhs-result-low  .nhs-result-pct { color: #007F3B; }
  .nhs-result-mod  .nhs-result-pct { color: #ED8B00; }
  .nhs-result-high .nhs-result-pct { color: #DA291C; }

  .nhs-result-tier {
    font-size: 1.1rem; font-weight: 700; margin: 6px 0 0;
  }
  .nhs-result-low  .nhs-result-tier { color: #007F3B; }
  .nhs-result-mod  .nhs-result-tier { color: #ED8B00; }
  .nhs-result-high .nhs-result-tier { color: #DA291C; }

  /* ── NHS warning notice ── */
  .nhs-warning {
    background: #fff9c4;
    border-left: 6px solid #FFB81C;
    padding: 12px 16px;
    font-size: 0.82rem;
    color: #212b32;
    line-height: 1.6;
    margin-top: 20px;
  }

  /* ── NHS green button override ── */
  div[data-testid="stButton"] button[kind="primary"] {
    background-color: #007F3B !important;
    border-color: #007F3B !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 4px !important;
    font-size: 1rem !important;
    padding: 10px 16px !important;
  }
  div[data-testid="stButton"] button[kind="primary"]:hover {
    background-color: #00602D !important;
    border-color: #00602D !important;
  }

  /* ── Gap between header and tabs ── */
  [data-testid="stTabs"] { margin-top: 20px !important; }

  /* ── Tab styling — remove ink, use solid block ── */
  [data-baseweb="tab-highlight"] { display: none !important; }
  [data-baseweb="tab-border"]    { display: none !important; }

  [data-baseweb="tab"] {
    font-size: 0.95rem !important;
    font-weight: 400 !important;
    color: #212b32 !important;
    background: #d8dde0 !important;
    border-radius: 0 !important;
    padding: 10px 18px !important;
    margin-right: 4px !important;
    border: none !important;
  }
  [data-baseweb="tab"]:hover {
    background: #aeb7bd !important;
    color: #212b32 !important;
  }
  [aria-selected="true"][data-baseweb="tab"] {
    background: #005EB8 !important;
    color: #ffffff !important;
    font-weight: 700 !important;
  }

  /* ── Placeholder text ── */
  .nhs-placeholder {
    color: #4c6272;
    font-size: 0.9rem;
    margin-top: 32px;
    line-height: 1.6;
  }
</style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_all():
    models, feat_cols = {}, {}
    for key in ["1yr", "5yr", "esrd_5yr", "esrd_10yr"]:
        models[key] = {}
        for clf in ["lr", "rf", "xgb", "lgbm"]:
            models[key][clf] = joblib.load(os.path.join(MODEL_DIR, f"{key}_{clf}.joblib"))
    feat_cols = joblib.load(os.path.join(MODEL_DIR, "feature_cols.joblib"))
    return models, feat_cols

try:
    MODELS, FEAT_COLS = load_all()
except FileNotFoundError:
    st.error("Model files not found. Run `python src/app/00_save_models.py` first.")
    st.stop()

# Lookup tables
LN_CLASS_OPT = {
    "Class I":1,"Class II":2,"Class III":3,"Class IV":4,"Class V":5,
    "Class III+V":6,"Class IV+V":7,"Class II+V":8,"Class VI":9,"Other":10,
}
ETHNICITY_OPT = {
    "White":1,"Black":2,"South Asian":3,
    "East Asian":4,"Other":5,"Not stated / Unknown":6,
}
BIOPSY_REASON = {
    "New presentation":1,"Relapse":2,
    "Non-response / partial response":3,
    "Pre-pregnancy or drug-switch assessment":4,
}
SUBEP_OPT = {
    "No deposits":0,"Small or rare deposits":1,
    "Large or conspicuous deposits":2,"No glomeruli on EM":3,
}

# Risk gauge HTML
def risk_gauge_html(p, analysis):
    t      = YOUDEN_THRESH[analysis]
    lo_pct = t * 0.60 * 100
    hi_pct = t        * 100
    pt_pct = min(max(p * 100, 1.5), 98.5)   # keep marker visible

    _, css = risk_tier(p, analysis)
    action = {
        "result-low":  "Routine follow-up is appropriate. Review at next scheduled clinic visit.",
        "result-mod":  "Consider more frequent monitoring. Discuss review interval with clinical team.",
        "result-high": "Intensified surveillance recommended. Consider early review and treatment discussion.",
    }[css]
    cls = css.replace("result-", "clinical-")

    return f"""
<div style="margin:12px 0 6px 0;">
  <p style="font-size:0.75rem;color:#4c6272;margin:0 0 6px 0;font-weight:600;
     letter-spacing:0.02em;text-transform:uppercase;">Risk spectrum</p>
  <div style="position:relative;height:18px;border-radius:3px;
    background:linear-gradient(to right,
      #007F3B 0%,#007F3B {lo_pct:.1f}%,
      #ED8B00 {lo_pct:.1f}%,#ED8B00 {hi_pct:.1f}%,
      #DA291C {hi_pct:.1f}%,#DA291C 100%);">
    <div style="position:absolute;left:{pt_pct:.1f}%;top:-5px;
      transform:translateX(-50%);width:3px;height:28px;
      background:#212b32;border-radius:2px;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;
    margin-top:4px;font-size:0.68rem;color:#6c757d;">
    <span>Low</span><span>Moderate</span><span>High</span>
  </div>
</div>
<div class="{cls}" style="margin-top:14px;">
  {action}
</div>"""

# Core helpers
def risk_tier(p, key):
    t = YOUDEN_THRESH[key]
    if p < t * 0.60:  return "Low risk",      "result-low"
    if p < t:         return "Moderate risk",  "result-mod"
    return              "High risk",           "result-high"

def predict_best(analysis, row_dict):
    df_row = pd.DataFrame([row_dict])[FEAT_COLS[analysis]]
    probs  = [float(MODELS[analysis][clf].predict_proba(df_row)[0, 1])
              for clf in ["lr", "rf", "xgb", "lgbm"]]
    p = float(np.mean(probs))
    return p, None, df_row

def show_result(p, _clf_key, df_row, analysis):
    tier, css = risk_tier(p, analysis)
    nhs_css = css.replace("result-", "nhs-result-")

    st.markdown(f"""
    <div class="{nhs_css}">
      <p class="nhs-result-pct">{p*100:.1f}%</p>
      <p class="nhs-result-tier">{tier}</p>
    </div>""", unsafe_allow_html=True)

    st.markdown(risk_gauge_html(p, analysis), unsafe_allow_html=True)

    st.markdown(
        '<div class="nhs-warning">'
        '<strong>Important:</strong> This is a research tool based on data from a single centre '
        'and has not been independently validated. It does not replace clinical judgement.'
        '</div>',
        unsafe_allow_html=True,
    )

# Page header

st.markdown("""
<div class="nhs-header">
  <p class="nhs-header-title">Lupus Nephritis Risk Calculator</p>
  <p class="nhs-header-sub">Imperial College London · Research tool</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "  1-Year Flare  ", "  5-Year Flare  ",
    "  Kidney failure (5-Year)  ", "  Kidney failure (10-Year)  ",
])

# TAB 1 — 1-Year Flare

with tab1:
    st.markdown(
        '<p class="nhs-intro">Estimates the likelihood of lupus nephritis flare '
        'within one year of this biopsy. Based on data from 430 patients.</p>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([5, 4], gap="large")

    with left:
        st.markdown('<p class="nhs-section">Patient details</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        age_1 = c1.number_input("Age at biopsy", 15, 90, 40, key="age1",
                                 help="Years")
        ln_1  = c2.selectbox("LN class", list(LN_CLASS_OPT.keys()), index=3, key="ln1")
        eth_1 = c1.selectbox("Ethnicity", list(ETHNICITY_OPT.keys()), key="eth1")
        c4_1  = c2.number_input("C4 at biopsy (g/L)", 0.00, 2.00, 0.15, step=0.01, key="c41")

        st.markdown('<p class="nhs-section">Biopsy findings</p>', unsafe_allow_html=True)
        ch_1 = st.slider("Chronic glomeruli (%)",  0, 100, 20, key="ch1")
        ac_1 = st.slider("Active glomeruli (%)",   0, 100, 30, key="ac1")
        cr_1 = st.slider("Crescents (%)",          0, 100,  5, key="cr1")
        ne_1 = st.slider("Necrosis (%)",           0, 100,  5, key="ne1")

        st.markdown('<p class="nhs-section">Blood results</p>', unsafe_allow_html=True)
        prot_raw_1 = st.number_input(
            "Proteinuria — uPCR (mg/mmol)", 0.0, 5000.0, 150.0, step=10.0, key="prot1",
            help="Entered as raw uPCR; log-transformed internally before prediction.",
        )

        st.write("")
        if st.button("Calculate", type="primary", key="btn1", use_container_width=True):
            row = {
                "% chronic gloms(%of total)": float(ch_1),
                "%gloms with necrosis": float(ne_1),
                "Age at biopsy": float(age_1),
                "Proteinuria at biopsy (uPCR, log)": float(np.log1p(prot_raw_1)),
                "Class coded 1=I 2=II 3=III 4=IV 5=V 6=III+V 7=IV+V 8=II+V 9=VI 10=other": float(LN_CLASS_OPT[ln_1]),
                "Ethnicity 1=white 2=black 3=asian (south) 4=asian (east) 5=other 6=not stated/unknown/any other mixed": float(ETHNICITY_OPT[eth_1]),
                "% active gloms (%of those not globally sclerosed)": float(ac_1),
                "%gloms with crescents": float(cr_1),
                "C4 at biopsy": float(c4_1),
            }
            p, clf_key, df_row = predict_best("1yr", row)
            st.session_state["res1"] = (p, clf_key, df_row)

    with right:
        if "res1" in st.session_state:
            show_result(*st.session_state["res1"], "1yr")
        else:
            st.markdown('<p class="nhs-placeholder">Complete the fields on the left and press <strong>Calculate</strong> to see the risk estimate.</p>', unsafe_allow_html=True)

# TAB 2 — 5-Year Flare

with tab2:
    st.markdown(
        '<p class="nhs-intro">Estimates the likelihood of lupus nephritis flare '
        'within five years of this biopsy. Based on data from 356 patients.</p>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([5, 4], gap="large")

    with left:
        st.markdown('<p class="nhs-section">Patient details</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        age_5    = c1.number_input("Age at biopsy", 15, 90, 40, key="age5")
        ln_5     = c2.selectbox("LN class", list(LN_CLASS_OPT.keys()), index=3, key="ln5")
        eth_5    = c1.selectbox("Ethnicity", list(ETHNICITY_OPT.keys()), key="eth5")
        egfr_5   = c2.number_input("eGFR (mL/min/1.73m²)", 5, 150, 60, key="egfr5")
        reason_5 = st.selectbox("Reason for biopsy", list(BIOPSY_REASON.keys()), key="reason5")

        c3, c4 = st.columns(2)
        dsdna_5 = c3.radio("dsDNA / SM / APL ever positive?", ["Yes", "No"], key="dsdna5")
        cyclo_5 = c4.radio("Prior cyclophosphamide?",         ["Yes", "No"], key="cyclo5")

        st.markdown('<p class="nhs-section">Biopsy findings</p>', unsafe_allow_html=True)
        ch_5 = st.slider("Chronic glomeruli (%)",   0, 100, 20, key="ch5")
        ac_5 = st.slider("Active glomeruli (%)",    0, 100, 30, key="ac5")
        sc_5 = st.slider("Sclerosed glomeruli (%)", 0, 100, 10, key="sc5")
        ne_5 = st.slider("Necrosis (%)",            0, 100,  5, key="ne5")

        st.write("")
        if st.button("Calculate", type="primary", key="btn5", use_container_width=True):
            row = {
                "% chronic gloms(%of total)": float(ch_5),
                "%gloms with necrosis": float(ne_5),
                "Class coded 1=I 2=II 3=III 4=IV 5=V 6=III+V 7=IV+V 8=II+V 9=VI 10=other": float(LN_CLASS_OPT[ln_5]),
                "% active gloms (%of those not globally sclerosed)": float(ac_5),
                "dsDNA or SM or APL ever positive(1=yes 0=no)": 1.0 if dsdna_5 == "Yes" else 0.0,
                "Prev exposure to cyclo (for Rx comparison) - related to the 'use this biopsy for this patient' biopsy": 1.0 if cyclo_5 == "Yes" else 0.0,
                "% sclerosed gloms": float(sc_5),
                "CKD epi formula without ethnicity": float(egfr_5),
                "Reason for biopsy 1=new pres LN 2=relapse 3=non-response/partial response, incl on-going proteinuria 4=pre-pregnancy or Ax if drug switch/stop appropriate": float(BIOPSY_REASON[reason_5]),
                "Ethnicity 1=white 2=black 3=asian (south) 4=asian (east) 5=other 6=not stated/unknown/any other mixed": float(ETHNICITY_OPT[eth_5]),
                "Age at biopsy": float(age_5),
            }
            p, clf_key, df_row = predict_best("5yr", row)
            st.session_state["res5"] = (p, clf_key, df_row)

    with right:
        if "res5" in st.session_state:
            show_result(*st.session_state["res5"], "5yr")
        else:
            st.markdown('<p class="nhs-placeholder">Complete the fields on the left and press <strong>Calculate</strong> to see the risk estimate.</p>', unsafe_allow_html=True)

# TAB 3 — ESRD 5-Year

with tab3:
    st.markdown(
        '<p class="nhs-intro">Estimates the likelihood of kidney failure '
        '(creatinine doubling, dialysis, or transplant) within five years of this biopsy. '
        'Based on data from 796 patients.</p>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([5, 4], gap="large")

    with left:
        st.markdown('<p class="nhs-section">Blood results</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        creat_e5 = c1.number_input("Creatinine at biopsy (mg/dL)", 0.3, 20.0, 1.0, step=0.1, key="creat_e5")
        egfr_e5  = c2.number_input("eGFR (mL/min/1.73m²)", 5, 150, 60, key="egfr_e5")

        st.markdown('<p class="nhs-section">Biopsy findings</p>', unsafe_allow_html=True)
        ifta_e5  = st.slider("IFTA (%)", 0, 100, 10, key="ifta_e5")
        ch_e5    = st.slider("Chronic glomeruli (%)", 0, 100, 15, key="ch_e5")
        subep_e5 = st.selectbox("Subepithelial deposits (electron microscopy)",
                                list(SUBEP_OPT.keys()), key="subep_e5")

        st.write("")
        if st.button("Calculate", type="primary", key="btn_e5", use_container_width=True):
            row = {
                "Creatinine at biopsy": float(creat_e5),
                "%IFTA ": float(ifta_e5),
                "CKD epi formula without ethnicity": float(egfr_e5),
                "% chronic gloms(%of total)": float(ch_e5),
                "Subepithelial deposit category (0=no deposits, 1=small/rare deposits, 2=large/conspicuous deposits, 3=no gloms on EM)": float(SUBEP_OPT[subep_e5]),
            }
            p, clf_key, df_row = predict_best("esrd_5yr", row)
            st.session_state["res_e5"] = (p, clf_key, df_row)

    with right:
        if "res_e5" in st.session_state:
            show_result(*st.session_state["res_e5"], "esrd_5yr")
        else:
            st.markdown('<p class="nhs-placeholder">Complete the fields on the left and press <strong>Calculate</strong> to see the risk estimate.</p>', unsafe_allow_html=True)

# TAB 4 — ESRD 10-Year

with tab4:
    st.markdown(
        '<p class="nhs-intro">Estimates the likelihood of kidney failure '
        'within ten years of this biopsy. Based on data from 796 patients.</p>',
        unsafe_allow_html=True,
    )
    left, right = st.columns([5, 4], gap="large")

    with left:
        st.markdown('<p class="nhs-section">Patient details</p>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        age_e10    = c1.number_input("Age at biopsy", 15, 90, 40, key="age_e10")
        ln_e10     = c2.selectbox("LN class", list(LN_CLASS_OPT.keys()), index=3, key="ln_e10")
        gender_e10 = c1.radio("Sex", ["Female", "Male"], key="gender_e10")

        st.markdown('<p class="nhs-section">Blood results</p>', unsafe_allow_html=True)
        c3, c4 = st.columns(2)
        creat_e10 = c3.number_input("Creatinine at biopsy (mg/dL)", 0.3, 20.0, 1.0, step=0.1, key="creat_e10")
        egfr_e10  = c4.number_input("eGFR (mL/min/1.73m²)", 5, 150, 60, key="egfr_e10")
        c3_e10    = c3.number_input("C3 at biopsy (g/L)", 0.0, 4.0, 0.9, step=0.05, key="c3_e10",
                                     help="Normal range 0.7–1.7 g/L")
        c4low_e10 = c4.radio("C4 low?", ["No", "Yes"], key="c4low_e10")

        st.markdown('<p class="nhs-section">Biopsy findings</p>', unsafe_allow_html=True)
        ifta_e10 = st.slider("IFTA (%)", 0, 100, 10, key="ifta_e10")
        ch_e10   = st.slider("Chronic glomeruli (%)", 0, 100, 15, key="ch_e10")

        c5, c6 = st.columns(2)
        n_sc_e10   = c5.number_input("Globally sclerosed gloms (n)", 0, 60, 2, key="nsc_e10")
        n_cr_e10   = c6.number_input("Gloms with crescents (n)", 0, 30, 0, key="ncr_e10")
        bx_num_e10 = c5.number_input("Biopsy number (1 = index)", 1, 10, 1, key="bxnum_e10")
        capigm_e10 = c6.number_input("Cap wall IgM (0–3 scale)", 0, 3, 0, key="capigm_e10")

        st.markdown('<p class="nhs-section">Additional findings</p>', unsafe_allow_html=True)
        c7, c8 = st.columns(2)
        cresc_e10 = c7.radio("Crescents present?",     ["No", "Yes"], key="cresc_e10")
        tma_e10   = c8.radio("TMA present?",           ["No", "Yes"], key="tma_e10")
        cyclo_e10 = c7.radio("Prior cyclophosphamide?",["No", "Yes"], key="cyclo_e10")
        subep_e10 = st.selectbox("Subepithelial deposits (electron microscopy)",
                                 list(SUBEP_OPT.keys()), key="subep_e10")

        st.write("")
        if st.button("Calculate", type="primary", key="btn_e10", use_container_width=True):
            row = {
                "Age at biopsy": float(age_e10),
                "%IFTA ": float(ifta_e10),
                "Creatinine at biopsy": float(creat_e10),
                "% chronic gloms(%of total)": float(ch_e10),
                "Crescents (Yes=1, No=0)": 1.0 if cresc_e10 == "Yes" else 0.0,
                "CKD epi formula without ethnicity": float(egfr_e10),
                "TMA (Yes=1, No=0)": 1.0 if tma_e10 == "Yes" else 0.0,
                "Class coded 1=I 2=II 3=III 4=IV 5=V 6=III+V 7=IV+V 8=II+V 9=VI 10=other": float(LN_CLASS_OPT[ln_e10]),
                "C3 at biopsy (normal range 0.7-1.7)": float(c3_e10),
                "C4 low (for range 0.15-0.54)": 1.0 if c4low_e10 == "Yes" else 0.0,
                "Prev exposure to cyclo (for Rx comparison) - related to the 'use this biopsy for this patient' biopsy": 1.0 if cyclo_e10 == "Yes" else 0.0,
                "Gender (1=male, 2=female)": 1.0 if gender_e10 == "Male" else 2.0,
                "Subepithelial deposit category (0=no deposits, 1=small/rare deposits, 2=large/conspicuous deposits, 3=no gloms on EM)": float(SUBEP_OPT[subep_e10]),
                "Cap wall IgM": float(capigm_e10),
                "No. globally sclerosed gloms ": float(n_sc_e10),
                "Biopsy number for patient": float(bx_num_e10),
                "No. gloms with crescents": float(n_cr_e10),
            }
            p, clf_key, df_row = predict_best("esrd_10yr", row)
            st.session_state["res_e10"] = (p, clf_key, df_row)

    with right:
        if "res_e10" in st.session_state:
            show_result(*st.session_state["res_e10"], "esrd_10yr")
        else:
            st.markdown('<p class="nhs-placeholder">Complete the fields on the left and press <strong>Calculate</strong> to see the risk estimate.</p>', unsafe_allow_html=True)

# Footer
st.markdown("""
<hr style="border:none;border-top:1px solid #d8dde0;margin-top:40px;">
<p style="font-size:0.78rem;color:#4c6272;line-height:1.6;">
  Imperial College London · Lupus Nephritis Prediction Project ·
  For research purposes only · Not validated for clinical use
</p>
""", unsafe_allow_html=True)
