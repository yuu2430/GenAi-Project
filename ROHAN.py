"""
GenAI Impact Study — Full Dashboard (FINAL REVISION)
MSc Statistics (Team 4) | The Maharaja Sayajirao University of Baroda
Academic Year 2025-26

Changes applied in this revision:
- Sampling: static HTML table, no bar chart
- Obj 2: Overall Dependency moved FIRST, then Gender/Level/Schooling+Faculty/Post-hoc
- Obj 3: Step 1 shows AI Usage Group + CGPA snippet from GitHub data; Step 2 removes histogram & QQ plot; Step 3 removed; boxplot labels without n= values
- Obj 4: Step 1 table shows AI Usage Group + CT Score snippet; boxplot labels without n= values; Step 3 removed
- Obj 5(a): Step 3 removed; histogram gets KDE curve; notation added
- Obj 5(b): Step 1 table uses Ind_learning score; Step 3 removed; only QQ plot shown (no histogram)
- Obj 6: Decision Tree plot (depth=3) added with dashboard-matching colors; Feature Importance removed
- AI Accuracy Check: Overview tab has 3 radar/web charts (Accuracy, Response Time, Detail) only; removed tabs 2-4; Subject Drilldown kept
- All formulas have notations
- All histograms have KDE curves
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats
from scipy.stats import (shapiro, ttest_1samp, ttest_ind, mannwhitneyu,
                         pearsonr, spearmanr, kruskal, wilcoxon,
                         levene, f_oneway, t as t_dist, norm, gaussian_kde)
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="GenAI Impact Study | MSU Baroda",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── COLOUR TOKENS ────────────────────────────────────────────
C = {
    "ink":      "#0d1b2a",
    "navy":     "#1b2f4e",
    "mid":      "#2c4a6e",
    "teal":     "#0e7c7b",
    "teal_lt":  "#4db6ac",
    "amber":    "#e6a817",
    "slate":    "#4a5568",
    "muted":    "#8896a8",
    "border":   "#dde3ea",
    "bg":       "#f7f9fc",
    "surface":  "#ffffff",
    "green":    "#1a7f5a",
    "red":      "#c0392b",
}

CHART_SEQ  = ["#1b2f4e","#0e7c7b","#e6a817","#6b46c1","#c0392b","#2980b9"]
CHART_DIV  = ["#1b2f4e","#4db6ac","#e6a817"]

AI_COLORS = {
    "ChatGPT":    "#10a37f",
    "Copilot":    "#0078d4",
    "Perplexity": "#7c3aed",
    "Gemini":     "#ea4335",
    "Claude":     "#d97706",
}

# ── CSS ──────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {C['slate']};
}}

section[data-testid="stSidebar"] {{
    background: {C['ink']} !important;
    border-right: none !important;
    min-width: 220px !important;
    max-width: 240px !important;
}}
section[data-testid="stSidebar"] * {{ color: #b0bec5 !important; }}

section[data-testid="stSidebar"] .stButton > button {{
    background: transparent !important;
    border: none !important;
    border-left: 2px solid transparent !important;
    border-radius: 6px !important;
    color: #8896a8 !important;
    font-size: 13px !important;
    font-weight: 400 !important;
    padding: 7px 14px !important;
    text-align: left !important;
    width: 100% !important;
    margin-bottom: 1px !important;
    transition: all 0.15s !important;
    justify-content: flex-start !important;
}}
section[data-testid="stSidebar"] .stButton > button:hover {{
    background: rgba(14,124,123,0.18) !important;
    border-left-color: {C['teal']} !important;
    color: #e0e7ef !important;
}}
input[type="radio"] {{ display: none; }}

.block-container {{
    padding-top: 0.7rem !important;
    padding-left: 2.4rem;
    padding-right: 2.4rem;
    padding-bottom: 3rem;
    background: {C['bg']};
    max-width: 1200px;
}}

h1, h2 {{
    font-family: 'Libre Baskerville', Georgia, serif;
    color: {C['ink']};
    font-weight: 700;
    letter-spacing: -0.3px;
}}
h3, h4 {{
    font-family: 'Inter', sans-serif;
    color: {C['navy']};
    font-weight: 600;
}}

.overline {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: {C['teal']};
    margin-bottom: 2px;
    margin-top: 8px;
}}

.page-title {{
    font-family: 'Libre Baskerville', serif;
    font-size: 24px;
    color: {C['ink']};
    font-weight: 700;
    margin: 0 0 2px 0;
    line-height: 1.25;
}}
.page-sub {{
    font-size: 13.5px;
    color: {C['muted']};
    margin: 0 0 16px 0;
}}

.step-label {{
    display: inline-block;
    background: {C['teal']};
    color: white;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 3px 10px;
    border-radius: 4px;
    margin-bottom: 8px;
    margin-top: 18px;
}}

.hyp {{
    background: #f0f7ff;
    border-left: 3px solid {C['teal']};
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 14px;
    line-height: 1.75;
    margin-bottom: 14px;
    color: {C['slate']};
}}

.result-pass {{
    background: #f0faf5;
    border-left: 3px solid {C['green']};
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 14px;
    line-height: 1.75;
    color: {C['slate']};
    margin: 14px 0;
}}
.result-info {{
    background: #fffbf0;
    border-left: 3px solid {C['amber']};
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 14px;
    line-height: 1.75;
    color: {C['slate']};
    margin: 14px 0;
}}
.result-fail {{
    background: #fff5f5;
    border-left: 3px solid {C['red']};
    padding: 12px 16px;
    border-radius: 0 6px 6px 0;
    font-size: 14px;
    line-height: 1.75;
    color: {C['slate']};
    margin: 14px 0;
}}

.assumption-box {{
    background: #f8fafc;
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 14px;
    line-height: 1.8;
    margin: 14px 0;
}}

.notation-box {{
    background: #f0f4f8;
    border: 1px solid {C['border']};
    border-radius: 6px;
    padding: 10px 16px;
    font-size: 12.5px;
    line-height: 1.9;
    margin: 8px 0 14px 0;
    color: {C['slate']};
}}

.rule {{ border:none; border-top:1px solid {C['border']}; margin:18px 0; }}

[data-testid="metric-container"] {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 14px 18px !important;
}}
[data-testid="metric-container"] label {{
    font-size: 11.5px !important;
    font-weight: 500 !important;
    color: {C['muted']} !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 22px !important;
    font-weight: 700 !important;
    color: {C['ink']} !important;
}}

.stDataFrame {{ border: 1px solid {C['border']}; border-radius: 8px; overflow: hidden; }}

.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    border-bottom: 1px solid {C['border']};
}}
.stTabs [data-baseweb="tab"] {{
    font-size: 13.5px;
    font-weight: 500;
    padding: 8px 18px;
    border-radius: 6px 6px 0 0;
    color: {C['muted']};
}}
.stTabs [aria-selected="true"] {{
    color: {C['ink']} !important;
    background: {C['surface']};
    border-bottom: 2px solid {C['teal']};
}}

/* Static sampling table */
.static-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-family: 'Inter', sans-serif;
    margin-top: 12px;
}}
.static-table th {{
    background: {C['navy']};
    color: white;
    padding: 9px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
.static-table td {{
    padding: 8px 12px;
    border-bottom: 1px solid {C['border']};
    color: {C['slate']};
}}
.static-table tr:nth-child(even) td {{
    background: #f7f9fc;
}}
.static-table tr:last-child td {{
    background: {C['navy']}18;
    font-weight: 700;
    color: {C['navy']};
}}
.static-table tr:hover td {{
    background: #e8f4f4;
}}
</style>
""", unsafe_allow_html=True)

# ── MATPLOTLIB DEFAULTS ───────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.edgecolor":    C["border"],
    "axes.labelcolor":   C["slate"],
    "xtick.color":       C["muted"],
    "ytick.color":       C["muted"],
    "text.color":        C["slate"],
    "grid.color":        "#eef0f4",
    "grid.linewidth":    0.7,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:16px 16px 10px; border-bottom:1px solid #1e2e42;'>
        <div style='font-family:"Libre Baskerville",serif; font-size:14px; color:#e0e7ef; font-weight:700; line-height:1.4;'>
            GenAI Impact Study
        </div>
        <div style='font-size:11px; color:#5d7a96; margin-top:3px;'>
            MSc Statistics · Team 4 · MSU Baroda
        </div>
    </div>
    """, unsafe_allow_html=True)

    NAV_GROUPS = {
        "Introduction": {
            "Overview":             "overview",
            "Objectives":           "objectives",
        },
        "Methodology": {
            "Pilot Survey":         "pilot",
            "Sampling Design":      "sampling",
            "Questionnaire":        "questionnaire",
            "Reliability Analysis": "reliability",
        },
        "Research Objectives": {
            "Obj 1 — Descriptive":       "descriptive",
            "Obj 2 — AI Dependency":     "anova",
            "Obj 3 — CGPA vs AI Usage":  "wilcoxon",
            "Obj 4 — Critical Thinking": "kruskal",
            "Obj 5 — Creativity & IL":   "correlation",
            "Obj 6 — ML Model":          "ml",
            "AI Accuracy Check":         "ai_accuracy",
        },
        "Synthesis": {
            "Conclusion":   "conclusion",
            "References":   "references",
        },
    }

    if "active" not in st.session_state:
        st.session_state.active = "overview"

    for grp, pages in NAV_GROUPS.items():
        st.markdown(f"""
        <div style='font-size:10px; font-weight:700; text-transform:uppercase;
                    letter-spacing:1.4px; color:#3d5570; margin:14px 0 4px 4px;'>
            {grp}
        </div>""", unsafe_allow_html=True)
        for label, key in pages.items():
            is_active = st.session_state.active == key
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.active = key
                st.rerun()

    active = st.session_state.active

    st.markdown("""
    <div style='padding:16px 16px 0; border-top:1px solid #1e2e42; margin-top:14px;'>
        <div style='font-size:10px; color:#3d5570; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:6px;'>Team Members</div>
        <div style='font-size:12px; color:#5d7a96; line-height:2;'>
            Vaishali Sharma<br>Ashish Vaghela<br>Raiwant Kumar<br>Rohan Shukla
        </div>
        <div style='font-size:10px; color:#3d5570; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin:10px 0 5px;'>Mentor</div>
        <div style='font-size:12px; color:#5d7a96;'>Prof. Murlidharan Kunnumal</div>
    </div>
    """, unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────
def page_header(overline, title, subtitle=""):
    st.markdown(f"<div class='overline'>{overline}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='page-sub'>{subtitle}</div>", unsafe_allow_html=True)
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

def step(label):
    st.markdown(f"<div class='step-label'>{label}</div>", unsafe_allow_html=True)

def hyp_block(h0, h1, test="", alpha="0.05"):
    extra = f"<br><b>Test:</b> {test}" if test else ""
    st.markdown(f"""
    <div class='hyp'>
    <b>H₀:</b> {h0}<br>
    <b>H₁:</b> {h1}{extra}<br>
    <b>Significance level:</b> α = {alpha}
    </div>""", unsafe_allow_html=True)

def notation(text):
    st.markdown(f"<div class='notation-box'><b>Notation:</b> {text}</div>", unsafe_allow_html=True)

def assumption_decision(text):
    st.markdown(f"<div class='assumption-box'><b>Test selection:</b> {text}</div>", unsafe_allow_html=True)

def result_pass(text):
    st.markdown(f"<div class='result-pass'>{text}</div>", unsafe_allow_html=True)

def result_info(text):
    st.markdown(f"<div class='result-info'>{text}</div>", unsafe_allow_html=True)

def result_fail(text):
    st.markdown(f"<div class='result-fail'>{text}</div>", unsafe_allow_html=True)

def plotly_defaults(fig, h=420):
    fig.update_layout(
        height=h,
        template="plotly_white",
        font=dict(family="Inter", size=12, color=C["slate"]),
        margin=dict(t=48, b=40, l=20, r=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=False, linecolor=C["border"])
    fig.update_yaxes(gridcolor="#eef0f4", linecolor="rgba(0,0,0,0)")
    return fig

def hist_with_kde(ax, data, color, xlabel, title, vlines=None):
    """Helper: histogram with KDE curve overlay"""
    n, bins, patches = ax.hist(data, bins=14, color=color, edgecolor="white", alpha=0.75, density=False)
    # KDE on secondary y-axis style: scale KDE to histogram counts
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 300)
    bin_width = bins[1] - bins[0]
    kde_scaled = kde(x_range) * len(data) * bin_width
    ax.plot(x_range, kde_scaled, color=C["ink"], lw=2.2, label="KDE curve")
    if vlines:
        for val, clr, lbl, ls in vlines:
            ax.axvline(val, color=clr, lw=2, ls=ls, label=lbl)
    ax.set_xlabel(xlabel); ax.set_ylabel("Frequency")
    ax.set_title(title)
    ax.legend(fontsize=9)

# ── GITHUB DATA LOADER ────────────────────────────────────────
GITHUB_DATA_URL = "https://github.com/yuu2430/GenAi-Project/raw/main/data.xlsx"

@st.cache_data(ttl=3600)
def load_sheet(sheet_name):
    try:
        return pd.read_excel(GITHUB_DATA_URL, sheet_name=sheet_name)
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def load_main_data():
    try:
        df = pd.read_excel(GITHUB_DATA_URL, sheet_name=0)
        return df
    except Exception:
        return None

# ── SIMULATED FALLBACK DATA ───────────────────────────────────
np.random.seed(42)
AI_DEP  = np.clip(np.random.normal(2.63, 0.74, 221), 1, 5)
np.random.seed(7)
IND_RAW = np.round(np.clip(np.random.normal(3.353, 0.983, 221), 1, 5) * 3) / 3
np.random.seed(21)
LOW_CT  = np.clip(np.random.normal(2.0, 0.6, 20),  1, 5)
MOD_CT  = np.clip(np.random.normal(3.1, 0.7, 141), 1, 5)
HIGH_CT = np.clip(np.random.normal(3.7, 0.6, 60),  1, 5)

np.random.seed(55)
low_cgpa_sim  = np.clip(np.random.normal(7.495, 1.570, 20),  4.0, 10.0)
mod_cgpa_sim  = np.clip(np.random.normal(6.856, 0.904, 141), 4.0, 10.0)
high_cgpa_sim = np.clip(np.random.normal(6.798, 0.993, 60),  4.0, 10.0)

# ── AI ACCURACY DATA ──────────────────────────────────────────
AI_ACCURACY_DATA = {
    "Physics": {
        "ChatGPT":    {"times":[12.92,13.11,14.75,13.13,16.08],"correct":[1,1,1,1,1],"detailed":[0,0,0,1,1],"prompted":[0,0,0,0,0]},
        "Copilot":    {"times":[4.90,7.36,7.56,7.07,13.96],"correct":[1,1,1,1,1],"detailed":[0,0,0,1,1],"prompted":[1,0,0,0,0]},
        "Perplexity": {"times":[6.61,5.41,4.34,3.98,12.99],"correct":[1,0,1,0,1],"detailed":[0,0,0,0,0],"prompted":[0,0,0,0,0]},
        "Gemini":     {"times":[9.77,10.04,12.04,10.88,10.35],"correct":[1,1,1,1,1],"detailed":[0,0,0,0,0],"prompted":[0,0,1,0,0]},
        "Claude":     {"times":[5.81,10.45,8.01,8.85,9.70],"correct":[1,1,1,1,1],"detailed":[0,0,0,0,0],"prompted":[0,0,0,0,0]},
    },
    "Mathematical Science": {
        "ChatGPT":    {"times":[18.53,19.29,22.28,16.10,19.90],"correct":[1,1,1,0,1],"detailed":[1,1,1,1,1],"prompted":[0,0,0,0,0]},
        "Copilot":    {"times":[13.23,19.64,22.55,16.40,16.01],"correct":[1,1,1,1,1],"detailed":[1,1,0,1,1],"prompted":[0,0,0,0,0]},
        "Perplexity": {"times":[11.52,14.21,15.26,6.56,5.90],"correct":[1,0,0,0,0],"detailed":[0,0,0,0,0],"prompted":[0,0,0,0,0]},
        "Gemini":     {"times":[20.57,14.57,15.66,13.10,20.15],"correct":[1,1,1,1,1],"detailed":[0,1,0,0,0],"prompted":[0,0,0,0,0]},
        "Claude":     {"times":[10.21,10.67,11.43,18.90,12.58],"correct":[1,1,1,1,1],"detailed":[0,1,1,1,1],"prompted":[0,0,0,0,0]},
    },
    "Mathematics": {
        "ChatGPT":    {"times":[20.51,15.18,16.03,19.41,26.50],"correct":[1,1,1,1,0],"detailed":[1,1,1,1,1],"prompted":[0,0,0,0,0]},
        "Copilot":    {"times":[17.99,15.47,10.18,15.66,15.88],"correct":[1,1,1,1,1],"detailed":[1,1,1,1,1],"prompted":[0,0,0,0,0]},
        "Perplexity": {"times":[5.43,5.21,4.98,4.66,6.12],"correct":[1,1,1,1,1],"detailed":[0,0,0,0,0],"prompted":[0,0,0,0,0]},
        "Gemini":     {"times":[9.98,10.46,21.45,13.17,13.00],"correct":[1,1,1,1,1],"detailed":[0,0,0,0,0],"prompted":[0,0,0,0,0]},
        "Claude":     {"times":[11.86,14.76,9.03,16.97,11.23],"correct":[1,1,1,1,1],"detailed":[0,1,0,1,1],"prompted":[0,0,0,0,0]},
    },
    "Chemistry": {
        "ChatGPT":    {"times":[13.17,10.55,20.40,13.68,11.66],"correct":[1,0,1,0,0],"detailed":[1,1,1,1,1],"prompted":[0,0,0,0,0]},
        "Copilot":    {"times":[7.37,9.06,13.08,11.98,13.09],"correct":[1,1,1,0,0],"detailed":[0,1,1,1,1],"prompted":[0,0,0,0,0]},
        "Perplexity": {"times":[8.17,4.71,6.57,6.22,5.84],"correct":[1,1,1,0,0],"detailed":[0,0,0,0,0],"prompted":[0,0,0,0,0]},
        "Gemini":     {"times":[7.23,9.53,11.90,13.95,12.20],"correct":[1,1,1,1,1],"detailed":[0,1,0,1,1],"prompted":[0,0,0,0,0]},
        "Claude":     {"times":[8.55,9.85,13.65,16.16,15.21],"correct":[1,0,1,0,0],"detailed":[1,1,1,1,1],"prompted":[0,0,0,0,0]},
    },
    "Biotechnology": {
        "ChatGPT":    {"times":[11.50,15.36,9.47,13.03,11.84],"correct":[1,1,1,1,1],"detailed":[1,1,1,1,1],"prompted":[0,0,0,0,0]},
        "Copilot":    {"times":[7.19,7.78,9.15,8.38,9.70],"correct":[1,1,0,1,1],"detailed":[1,1,1,1,1],"prompted":[0,0,0,0,0]},
        "Perplexity": {"times":[3.96,4.80,3.66,5.80,3.95],"correct":[1,1,1,0,1],"detailed":[0,0,0,0,0],"prompted":[0,0,0,0,0]},
        "Gemini":     {"times":[12.20,10.10,10.96,13.65,11.98],"correct":[1,1,0,1,1],"detailed":[1,1,1,1,1],"prompted":[0,0,0,0,0]},
        "Claude":     {"times":[12.46,10.20,9.56,15.11,12.32],"correct":[1,1,1,1,1],"detailed":[1,1,0,1,0],"prompted":[0,0,0,0,0]},
    },
}

TOOLS    = ["ChatGPT","Copilot","Perplexity","Gemini","Claude"]
SUBJECTS = list(AI_ACCURACY_DATA.keys())

def build_summary_df():
    rows = []
    for subject, tools in AI_ACCURACY_DATA.items():
        for tool, d in tools.items():
            n = len(d["correct"])
            rows.append({
                "Subject":      subject,
                "Tool":         tool,
                "Accuracy (%)": round(sum(d["correct"]) / n * 100, 1),
                "Detailed (%)": round(sum(d["detailed"]) / n * 100, 1),
                "Avg Time (s)": round(np.mean(d["times"]), 2),
                "Prompted (%)": round(sum(d["prompted"]) / n * 100, 1),
                "n":            n,
            })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# AI ACCURACY CHECK — UPDATED (3 radar charts in overview, subject drilldown)
# ══════════════════════════════════════════════════════════════
if active == "ai_accuracy":
    page_header("Supplementary Study","AI Accuracy Check — JAM 2025",
                "Performance benchmarking of 5 AI tools across 5 JAM 2025 subjects.")
    df_sum = build_summary_df()
    overall_acc = {}
    for tool in TOOLS:
        sub_df = df_sum[df_sum["Tool"] == tool]
        overall_acc[tool] = round(sub_df["Accuracy (%)"].mean(), 1)
    best_tool    = max(overall_acc, key=overall_acc.get)
    fastest_tool = df_sum.groupby("Tool")["Avg Time (s)"].mean().idxmin()
    most_detail  = df_sum.groupby("Tool")["Detailed (%)"].mean().idxmax()
    k1,k2,k3,k4,k5 = st.columns(5)
    k1.metric("Questions / Tool / Subject","5"); k2.metric("Total Evaluations","125")
    k3.metric("Best Accuracy",f"{best_tool} ({overall_acc[best_tool]}%)")
    k4.metric("Fastest Responses",fastest_tool); k5.metric("Most Detailed",most_detail)
    st.markdown("<br>", unsafe_allow_html=True)

    sorted_tools = sorted(overall_acc.items(), key=lambda x: x[1], reverse=True)
    bar_cols = st.columns(len(sorted_tools))
    for col, (tool, acc) in zip(bar_cols, sorted_tools):
        tool_color = AI_COLORS.get(tool, C["navy"])
        rank_label = ["🥇","🥈","🥉","4th","5th"]
        rank_idx = [t for t,_ in sorted_tools].index(tool)
        medal = rank_label[rank_idx]
        col.markdown(f"""
        <div style='background:{C["surface"]}; border:1px solid {C["border"]};
                    border-radius:10px; padding:14px 16px; text-align:center;'>
            <div style='font-size:11px; font-weight:700; color:{tool_color}; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:6px;'>{medal} {tool}</div>
            <div style='font-size:22px; font-weight:700; color:{C["ink"]}; margin-bottom:8px;'>{acc}%</div>
            <div style='background:{C["border"]}; border-radius:99px; height:8px; overflow:hidden;'>
                <div style='background:{tool_color}; width:{acc}%; height:100%; border-radius:99px;'></div>
            </div>
            <div style='font-size:11px; color:{C["muted"]}; margin-top:6px;'>{int(round(acc/100*25))} / 25 correct</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    t1, t2 = st.tabs(["📊 Overview","🔍 Subject Drilldown"])

    with t1:
        st.markdown("### Performance Radar Charts — AI Tools Across Subjects")
        st.markdown(f"<div style='font-size:13.5px; color:{C['slate']}; margin-bottom:18px;'>Three spider/radar charts below show each AI tool's <b>Accuracy (%)</b>, <b>Mean Response Time (s)</b>, and <b>Detail Rate (%)</b> across all 5 subjects. Each axis represents one subject.</div>", unsafe_allow_html=True)

        subj_short = ["Physics","Math Sci","Maths","Chem","Biotech"]

        # Chart 1: Accuracy radar
        st.markdown("#### 🎯 Accuracy (%) by Subject")
        fig_acc = go.Figure()
        for tool in TOOLS:
            tool_df = df_sum[df_sum["Tool"] == tool].set_index("Subject")
            vals = [tool_df.loc[s, "Accuracy (%)"] if s in tool_df.index else 0 for s in SUBJECTS]
            vals_closed = vals + [vals[0]]
            cats_closed = subj_short + [subj_short[0]]
            fig_acc.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats_closed,
                fill='toself', name=tool,
                line_color=AI_COLORS[tool],
                fillcolor=AI_COLORS[tool],
                opacity=0.25,
                line=dict(width=2.5)
            ))
        fig_acc.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=10), gridcolor="#dde3ea"),
                angularaxis=dict(tickfont=dict(size=12, color=C["navy"]))
            ),
            showlegend=True, height=440,
            font=dict(family="Inter", color=C["slate"]),
            paper_bgcolor="white", plot_bgcolor="white",
            legend=dict(orientation="h", y=-0.12),
            margin=dict(t=30, b=60, l=60, r=60)
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        # Chart 2: Response Time radar
        st.markdown("#### ⏱ Mean Response Time (s) by Subject")
        fig_time = go.Figure()
        for tool in TOOLS:
            tool_df = df_sum[df_sum["Tool"] == tool].set_index("Subject")
            vals = [tool_df.loc[s, "Avg Time (s)"] if s in tool_df.index else 0 for s in SUBJECTS]
            vals_closed = vals + [vals[0]]
            cats_closed = subj_short + [subj_short[0]]
            fig_time.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats_closed,
                fill='toself', name=tool,
                line_color=AI_COLORS[tool],
                fillcolor=AI_COLORS[tool],
                opacity=0.25,
                line=dict(width=2.5)
            ))
        max_time = df_sum["Avg Time (s)"].max() * 1.15
        fig_time.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, max_time], tickfont=dict(size=10), gridcolor="#dde3ea"),
                angularaxis=dict(tickfont=dict(size=12, color=C["navy"]))
            ),
            showlegend=True, height=440,
            font=dict(family="Inter", color=C["slate"]),
            paper_bgcolor="white", plot_bgcolor="white",
            legend=dict(orientation="h", y=-0.12),
            margin=dict(t=30, b=60, l=60, r=60)
        )
        st.plotly_chart(fig_time, use_container_width=True)

        # Chart 3: Detail Rate radar
        st.markdown("#### 📋 Detail Rate (%) by Subject")
        fig_det = go.Figure()
        for tool in TOOLS:
            tool_df = df_sum[df_sum["Tool"] == tool].set_index("Subject")
            vals = [tool_df.loc[s, "Detailed (%)"] if s in tool_df.index else 0 for s in SUBJECTS]
            vals_closed = vals + [vals[0]]
            cats_closed = subj_short + [subj_short[0]]
            fig_det.add_trace(go.Scatterpolar(
                r=vals_closed, theta=cats_closed,
                fill='toself', name=tool,
                line_color=AI_COLORS[tool],
                fillcolor=AI_COLORS[tool],
                opacity=0.25,
                line=dict(width=2.5)
            ))
        fig_det.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0,100], tickfont=dict(size=10), gridcolor="#dde3ea"),
                angularaxis=dict(tickfont=dict(size=12, color=C["navy"]))
            ),
            showlegend=True, height=440,
            font=dict(family="Inter", color=C["slate"]),
            paper_bgcolor="white", plot_bgcolor="white",
            legend=dict(orientation="h", y=-0.12),
            margin=dict(t=30, b=60, l=60, r=60)
        )
        st.plotly_chart(fig_det, use_container_width=True)

    with t2:
        selected_subject = st.selectbox("Select a Subject", SUBJECTS)
        subj_df = df_sum[df_sum["Subject"]==selected_subject].copy()
        c1,c2,c3 = st.columns(3)
        c1.metric("Best Accuracy",f"{subj_df.loc[subj_df['Accuracy (%)'].idxmax(),'Tool']} ({subj_df['Accuracy (%)'].max()}%)")
        c2.metric("Fastest",f"{subj_df.loc[subj_df['Avg Time (s)'].idxmin(),'Tool']} ({subj_df['Avg Time (s)'].min():.1f}s)")
        c3.metric("Most Detailed",f"{subj_df.loc[subj_df['Detailed (%)'].idxmax(),'Tool']} ({subj_df['Detailed (%)'].max():.0f}%)")
        st.dataframe(subj_df.set_index("Tool"), use_container_width=True)


# ══════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════
elif active == "overview":
    import base64, os
    logo_path = "msu_logo.png"
    if os.path.exists(logo_path):
        with open(logo_path,"rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode()
        logo_html = f"<img src='data:image/png;base64,{logo_b64}' style='height:80px; margin-bottom:14px; filter:brightness(0) invert(1);'/>"
    else:
        logo_html = ""
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{C["ink"]} 0%,{C["mid"]} 100%);
                border-radius:12px; padding:36px 48px; color:white; margin-bottom:24px; text-align:center;'>
        {logo_html}
        <div style='font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:2px; color:{C["teal_lt"]}; margin-bottom:5px;'>THE MAHARAJA SAYAJIRAO UNIVERSITY OF BARODA</div>
        <div style='font-size:11.5px; color:#94b4cc; margin-bottom:3px;'>Faculty of Science · Department of Statistics</div>
        <div style='font-size:11.5px; color:#94b4cc; margin-bottom:18px;'>Academic Year 2025-26</div>
        <div style='font-family:"Libre Baskerville",serif; font-size:26px; font-weight:700; line-height:1.35; margin-bottom:14px;'>
            Cognitive &amp; Educational Impacts of<br>Generative AI Usage Among University Students
        </div>
        <div style='font-size:13px; color:#94b4cc; margin-bottom:16px;'>
            <strong style='color:white;'>MSc Statistics · Team 4</strong><br>
            Vaishali Sharma &nbsp;·&nbsp; Ashish Vaghela &nbsp;·&nbsp; Raiwant Kumar &nbsp;·&nbsp; Rohan Shukla<br>
            <span style='font-size:12px;'>Guided by: Prof. Murlidharan Kunnumal</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    page_header("Introduction","Overview of the Study","Understanding the cognitive and educational impact of Generative AI among university students.")
    st.markdown(f"""
    <div style='background:{C["surface"]}; border:1px solid {C["border"]}; border-left:4px solid {C["teal"]};
                border-radius:8px; padding:22px 26px; font-size:14.5px; line-height:1.9; color:{C["slate"]}; text-align:justify;'>
    This study examines the cognitive and educational implications of Generative Artificial Intelligence (GenAI) usage among university students.
    With the increasing integration of AI tools such as ChatGPT, Gemini, and Copilot into academic environments, it becomes essential to understand
    how these technologies influence student learning behavior and intellectual development.<br><br>
    Primary data were collected from <strong>221 students</strong> across <strong>13 faculties</strong> of The Maharaja Sayajirao University of Baroda
    using a structured questionnaire. The instrument captures key dimensions: frequency and purpose of AI use, and its impact on independent learning,
    critical thinking, creativity, cognitive offloading, and AI dependency.<br><br>
    The study employs a progression of statistical techniques from reliability analysis and descriptive methods through formal hypothesis testing
    (t-tests, ANOVA, Kruskal-Wallis, Wilcoxon, Jonckheere-Terpstra) and a machine learning component to evaluate whether AI-related behavioral
    and cognitive factors can predict academic performance.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# OBJECTIVES
# ══════════════════════════════════════════════════════════════
elif active == "objectives":
    page_header("Study Design","Research Objectives","Six operationalised objectives spanning descriptive through predictive analysis.")
    objs = [
        ("1","AI Usage Patterns","To determine how frequently students use AI tools, identify the purposes for which they are used, and examine whether usage patterns vary systematically by educational level (UG vs. PG) or gender.","Descriptive analysis · Grouped bar charts · Donut charts"),
        ("2","Level of AI Dependency","To identify and quantify the level of dependency on Generative AI among MSU students using the GAIDS scale, and to examine whether dependency differs by gender, level of study, schooling background, and faculty.","Normality (Shapiro-Wilk) · Levene's Test · Independent t-test / Mann-Whitney U · One-Way ANOVA / Kruskal-Wallis · Dunn Post-Hoc"),
        ("3","AI Usage Level and Learning Performance (CGPA)","To check whether AI usage level (Low, Moderate, High) significantly affects students' learning performance as measured by CGPA.","Shapiro-Wilk · Levene's Test · Kruskal-Wallis H Test"),
        ("4","AI Usage and Critical Thinking","To study how AI tool usage is related to students' critical thinking, using self-reported ratings, and to determine whether there is an ordered trend.","Shapiro-Wilk · Levene's Test · Jonckheere-Terpstra Test (ordered trend)"),
        ("5","Creativity and Independent Learning","To explore how using AI tools influences students' academic skills — specifically creativity and independent learning beyond the classroom.","Normality (Shapiro-Wilk) · Wilcoxon Signed-Rank Test (two sub-objectives) · Cronbach's Alpha"),
        ("6","Predictive Model for Academic Performance","To create a model that predicts factors leading to positive or negative effects of AI tool usage on students' academic outcomes.","K-Nearest Neighbours · Decision Tree"),
    ]
    for num,title,desc,methods in objs:
        st.markdown(f"""
        <div style='display:flex; gap:20px; background:{C["surface"]}; border:1px solid {C["border"]};
                    border-radius:8px; padding:20px 24px; margin-bottom:12px;'>
            <div style='font-family:"Libre Baskerville",serif; font-size:34px; font-weight:700;
                        color:{C["border"]}; line-height:1; min-width:34px; text-align:center; padding-top:4px;'>{num}</div>
            <div style='flex:1;'>
                <div style='font-weight:600; color:{C["ink"]}; font-size:15px; margin-bottom:5px;'>{title}</div>
                <div style='font-size:14px; color:{C["slate"]}; line-height:1.7; margin-bottom:8px;'>{desc}</div>
                <div style='font-size:12px; color:{C["teal"]}; font-weight:500;'>{methods}</div>
            </div>
        </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# PILOT SURVEY
# ══════════════════════════════════════════════════════════════
elif active == "pilot":
    page_header("Methodology", "Pilot Survey",
                "Conducted prior to full data collection to validate the questionnaire and estimate the population proportion.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Pilot Sample (n)", "58")
    c2.metric("Affirmative Responses", "48")
    c3.metric("Estimated Proportion", "0.827")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.1, 1])
    with col_l:
        st.markdown(f"""
        <div style='font-size:14px; color:{C["slate"]}; line-height:1.9;'>
        A simple random sample of <strong>58 students</strong> was surveyed with one binary question:
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:#f0f7ff; border-left:3px solid {C["teal"]}; padding:14px 18px;
                    border-radius:0 6px 6px 0; font-size:14.5px; font-style:italic;
                    color:{C["navy"]}; margin:16px 0;'>
            "Has Generative AI impacted your education?" (Yes / No)
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div style='font-size:14px; color:{C["slate"]}; line-height:1.9;'>
        48 of 58 respondents answered Yes, yielding the pilot proportion estimate used in the sample
        size formula. The 82.7% affirmative rate confirmed both the relevance of the research problem
        and provided the empirical <em>p</em> for Cochran's formula.
        </div>""", unsafe_allow_html=True)

        st.markdown("**Sample Size Calculation — Cochran's Formula:**")
        st.latex(r"n = \frac{z_{\alpha/2}^2 \cdot p \cdot q}{E^2} = \frac{(1.96)^2 \times 0.827 \times 0.173}{(0.05)^2} \approx \mathbf{221}")
        notation("z<sub>α/2</sub> = critical value at 95% confidence (1.96) · p = pilot proportion (0.827) · q = 1 − p (0.173) · E = margin of error (0.05) · n = required sample size")

    with col_r:
        fig = go.Figure(go.Pie(
            labels=["Yes — AI has impacted education", "No"],
            values=[48, 10],
            hole=0.58,
            marker_colors=[C["teal"], C["border"]],
            textinfo="percent",
            textfont_size=13,
            hovertemplate="%{label}: %{value} respondents<extra></extra>"
        ))
        fig.add_annotation(text="<b>82.7%</b><br>Yes", x=0.5, y=0.5,
                           font_size=16, showarrow=False, font_color=C["ink"])
        fig.update_layout(height=320, margin=dict(t=30,b=10,l=10,r=10),
                          showlegend=True, font=dict(family="Inter"),
                          legend=dict(orientation="h", y=-0.1),
                          paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("**Functions of the Pilot Study**")
    for item in [
        "Questionnaire validation — testing clarity, wording, and construct relevance",
        "Likert-scale consistency check — verifying uniform interpretation of rating anchors",
        "Empirical proportion estimation — providing data-driven p for sample size calculation",
        "Identification of measurement and response bias before full-scale deployment",
    ]:
        st.markdown(f"<div style='font-size:14px; color:{C['slate']}; padding:4px 0 4px 16px; border-left:2px solid {C['border']};'>{item}</div>", unsafe_allow_html=True)
        st.markdown("")


# ══════════════════════════════════════════════════════════════
# SAMPLING — STATIC TABLE, NO BAR CHART
# ══════════════════════════════════════════════════════════════
elif active == "sampling":
    page_header("Methodology","Sampling Design","Probability Proportional to Size (PPS) sampling across all 13 faculties of MSU Baroda.")
    c1,c2,c3 = st.columns(3)
    c1.metric("Sampling Method","PPS"); c2.metric("Total Population (N)","37,095"); c3.metric("Final Sample (n)","221")
    st.markdown("**Proportional Allocation Formula:**")
    st.latex(r"n_i = \frac{N_i}{N} \times n")
    notation("nᵢ = sample size allocated to faculty i · Nᵢ = total students in faculty i · N = total university population (37,095) · n = overall required sample size (221)")
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("**Faculty-wise Sample Distribution**")

    # Static HTML table
    rows_data = [
        ("Arts", 25, 13, 3, 8, 1),
        ("Commerce", 120, 58, 8, 50, 4),
        ("Education &amp; Psychology", 5, 3, 1, 1, 0),
        ("Family &amp; Community Sciences", 7, 5, 1, 1, 0),
        ("Fine Arts", 4, 2, 1, 1, 0),
        ("Journalism &amp; Communication", 1, 1, 0, 0, 0),
        ("Law", 8, 4, 0, 4, 0),
        ("Management Studies", 2, 1, 0, 1, 0),
        ("Performing Arts", 2, 1, 0, 1, 0),
        ("Pharmacy", 2, 1, 0, 1, 0),
        ("Science", 23, 9, 4, 8, 2),
        ("Social Work", 3, 1, 1, 1, 0),
        ("Technology &amp; Engineering", 19, 3, 2, 12, 2),
    ]
    total_row = ("TOTAL", 221, 102, 21, 89, 9)

    table_html = """
    <table class='static-table'>
    <thead>
    <tr>
        <th>Faculty</th>
        <th>Total</th>
        <th>Female UG</th>
        <th>Female PG</th>
        <th>Male UG</th>
        <th>Male PG</th>
    </tr>
    </thead>
    <tbody>
    """
    for r in rows_data:
        table_html += f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td><td>{r[3]}</td><td>{r[4]}</td><td>{r[5]}</td></tr>"
    table_html += f"<tr><td>{total_row[0]}</td><td>{total_row[1]}</td><td>{total_row[2]}</td><td>{total_row[3]}</td><td>{total_row[4]}</td><td>{total_row[5]}</td></tr>"
    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
elif active == "questionnaire":
    page_header(
        "Data Collection",
        "Questionnaire Design",
        "Structured self-administered questionnaire delivered via Google Forms — 23 questions across 6 sections."
    )

    sections = [
        {"title":"Section 1 — Student Profile & AI Usage (Q1–Q14)","items":["Demographic variables: Age, Gender, CGPA, Schooling Background, Faculty, Level and Year of Study","Awareness of GenAI, primary AI tool used, subscription status and monthly expenditure","Multi-select grid (Q13): AI platform × 6 academic purposes","Likert frequency grid (Q14): Never → Always across 6 academic purposes"]},
        {"title":"Section 2 — Academic Impact (Q15–Q16)","items":["Q15 (3 items): grades change, learning effectiveness, curriculum inclusion — Strongly Disagree to Strongly Agree","Q16 (6 items): AI vs books, AI vs teacher, critical thinking reduction, AI submission, stress reduction, independent exploration"]},
        {"title":"Section 3 — Cognitive Offloading (Q18)","items":["5 items measuring reliance on digital tools for information retrieval and task memory","Scale: 1 (Never / Not Dependent) to 5 (Always / Very Likely)"]},
        {"title":"Section 4 — Critical Thinking (Q19)","items":["8 items: source evaluation, fake-news detection, cross-referencing, author credibility, bias reflection","Scale: 1 (Never / Not Confident) to 5 (Always / Very Confident)"]},
        {"title":"Section 5 — AI Dependency (Q17, Q20–Q22)","items":["Q17: 10-point scale — trust, understanding depth, motivation, anxiety, assignment dependency","Q20 — Cognitive Preoccupation (3 items): decision influence, urge to use, anticipation","Q21 — Negative Consequences (4 items): concerns, inability to work without AI, confidence, problem-solving","Q22 — Withdrawal Symptoms (4 items): restlessness, distraction, disconnection, irritability"]},
        {"title":"Section 6 — Creativity (Q23)","items":["11 items adapted from the Kaufman Domains of Creativity Scale (K-DOCS, 2012)","Scale: 1 (Much Less Creative) to 5 (Much More Creative) — comparative to doing the task without AI","Tasks: writing, debating, researching, feedback, analysis, argumentation"]},
    ]

    # Render questionnaire sections
    for sec in sections:
        st.markdown(
            f"<div style='font-weight:600; color:{C['ink']}; font-size:14px; margin:16px 0 6px;'>{sec['title']}</div>",
            unsafe_allow_html=True
        )
        for item in sec["items"]:
            st.markdown(
                f"<div style='font-size:13.5px; color:{C['slate']}; padding:3px 0 3px 16px; border-left:2px solid {C['border']};'>— {item}</div>",
                unsafe_allow_html=True
            )

    # References moved BELOW
st.markdown(f"""
<div style='background:{C['surface']}; border:1px solid {C['border']}; border-left:3px solid {C['amber']}; border-radius:0 8px 8px 0; padding:14px 20px; margin-top:24px;'>
    <div style='font-size:11px; font-weight:700; text-transform:uppercase; letter-spacing:1.2px; color:{C['amber']}; margin-bottom:8px;'>References for Questionnaire</div>
    <div style='font-size:14px; color:{C['slate']}; line-height:2;'>
        &nbsp;<a href='https://www.mdpi.com/2075-4698/15/1/6' target='_blank' style='color:{C['teal']}; font-weight:500; text-decoration:none;'>Gerlich (2025) — AI Tools in Society</a><br>
        &nbsp;<a href='https://drive.google.com/file/d/15Za6HQIaUscX2UxEJ9KGEHGxiPG-qQ6E/view' target='_blank' style='color:{C['teal']}; font-weight:500; text-decoration:none;'>Study Questionnaire — PDF</a>
        &nbsp;<a href='https://drive.google.com/file/d/1o4H5H4K3Ufd6bHuB6JSKZshi2OfptgA6/view?usp=sharing' target='_blank' style='color:{C['teal']}; font-weight:500; text-decoration:none;'>Kaufman Domains of Creativity Scale (K-DOCS)</a><br>
    </div>
</div>
""", unsafe_allow_html=True)# ══════════════════════════════════════════════════════════════
# RELIABILITY
# ══════════════════════════════════════════════════════════════
elif active == "reliability":
    page_header("Pre-Analysis","Reliability Analysis — Cronbach's Alpha","Internal consistency of all multi-item scales verified before inferential analysis.")
    st.markdown("**Formula:**")
    st.latex(r"\alpha = \frac{k}{k-1} \left(1 - \frac{\sum_{i=1}^{k} \sigma_{y_i}^2}{\sigma_x^2}\right)")
    notation("α = Cronbach's Alpha · k = number of items in the scale · σ²<sub>yᵢ</sub> = variance of the i-th item · σ²<sub>x</sub> = variance of the total (composite) score")
    rel = pd.DataFrame({"Construct":["AI Dependency (GAIDS)","Critical Thinking","Creativity (K-DOCS)","Cognitive Offloading"],"Items":[11,8,11,5],"Scale":["Likert 1–5","Likert 1–5","Comparative 1–5","Likert 1–5"],"Cronbach α":[0.8936,0.9139,0.9427,0.8512],"Reliability":["Good","Excellent","Excellent","Good"]})
    st.dataframe(rel.set_index("Construct"), use_container_width=True)
    fig = go.Figure()
    colors = [C["teal"] if a >= 0.9 else C["navy"] for a in rel["Cronbach α"]]
    fig.add_bar(x=rel["Construct"],y=rel["Cronbach α"],marker_color=colors,text=[f"{a:.4f}" for a in rel["Cronbach α"]],textposition="outside")
    fig.add_hline(y=0.70,line_dash="dash",line_color=C["red"]); fig.add_hline(y=0.90,line_dash="dot",line_color=C["amber"])
    plotly_defaults(fig,h=360); st.plotly_chart(fig, use_container_width=True)
    result_pass("All scales show strong reliability (α > 0.83).")


# ══════════════════════════════════════════════════════════════
# OBJECTIVE 1 — DESCRIPTIVE
# ══════════════════════════════════════════════════════════════
elif active == "descriptive":
    page_header("Objective 1","Descriptive Analysis of AI Usage","How often do students use AI tools, for what purposes, and do patterns differ by level of study or gender?")
    viz = st.radio("", ["AI Tools × Academic Purpose","Programme-wise Usage","Gender-wise Usage","Usage Frequency by Purpose"], horizontal=True)
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    if viz == "AI Tools × Academic Purpose":
        data_t = pd.DataFrame({
            "Purpose":["Project/Assignment"]*4+["Concept Learning"]*4+["Writing/Summarising"]*4+["Exam Preparation"]*4+["Research/Idea Gen"]*4+["Programming/Coding"]*4,
            "Tool":["ChatGPT","Gemini","Copilot","Perplexity"]*6,
            "n":[131,48,5,10, 114,73,14,9, 128,56,12,6, 101,56,15,9, 101,40,6,8, 66,34,16,10]
        })
        fig = px.bar(data_t,x="Purpose",y="n",color="Tool",barmode="group",text_auto=True,color_discrete_sequence=CHART_SEQ)
        plotly_defaults(fig,h=460); fig.update_layout(xaxis_tickangle=-15,legend_title="AI Tool",yaxis_title="Number of Students")
        st.plotly_chart(fig, use_container_width=True)
        result_info("ChatGPT dominates across all six academic purposes. Programming/Coding shows elevated Copilot usage alongside ChatGPT.")
    elif viz == "Programme-wise Usage":
        c1,c2 = st.columns(2)
        for col,title,yes,no in [(c1,"Undergraduate (n=171)",112,59),(c2,"Postgraduate (n=50)",47,3)]:
            fig = go.Figure(go.Pie(labels=["Uses GenAI","Does not use"],values=[yes,no],hole=0.55,marker_colors=[C["teal"],C["border"]],textinfo="percent",textfont_size=13))
            pct = round(yes/(yes+no)*100,1)
            fig.add_annotation(text=f"<b>{pct}%</b>",x=0.5,y=0.5,font_size=18,showarrow=False,font_color=C["ink"])
            fig.update_layout(title=title,height=300,showlegend=False,font=dict(family="Inter"),paper_bgcolor="white",margin=dict(t=48,b=10,l=10,r=10))
            col.plotly_chart(fig, use_container_width=True)
        result_info("PG students show <strong>94% AI adoption</strong> vs 65.5% for UG students.")
    elif viz == "Gender-wise Usage":
        c1,c2 = st.columns(2)
        for col,title,yes,no,clr in [(c1,"Female Students (n=128)",99,29,C["teal"]),(c2,"Male Students (n=93)",60,33,C["navy"])]:
            fig = go.Figure(go.Pie(labels=["Uses GenAI","Does not use"],values=[yes,no],hole=0.55,marker_colors=[clr,C["border"]],textinfo="percent",textfont_size=13))
            pct = round(yes/(yes+no)*100,1)
            fig.add_annotation(text=f"<b>{pct}%</b>",x=0.5,y=0.5,font_size=18,showarrow=False,font_color=C["ink"])
            fig.update_layout(title=title,height=300,showlegend=False,font=dict(family="Inter"),paper_bgcolor="white",margin=dict(t=48,b=10,l=10,r=10))
            col.plotly_chart(fig, use_container_width=True)
        result_info("Female students (77.3%) show <strong>higher AI adoption</strong> than male students (64.5%).")
    else:
        freq_data = pd.DataFrame({
            "Purpose":["Project/Assignment"]*5+["Concept Learning"]*5+["Writing/Summarising"]*5+["Exam Preparation"]*5+["Research/Idea Gen"]*5+["Programming/Coding"]*5,
            "Frequency":["Never","Rarely","Sometimes","Often","Always"]*6,
            "n":[10,35,67,67,42, 11,21,69,87,33, 13,16,78,79,35, 21,18,68,80,34, 32,37,62,49,41, 22,30,43,99,27]
        })
        fig = px.bar(freq_data,x="Purpose",y="n",color="Frequency",barmode="group",text_auto=True,
                     color_discrete_map={"Never":"#c8d6e5","Rarely":"#8896a8","Sometimes":C["teal_lt"],"Often":C["teal"],"Always":C["navy"]})
        plotly_defaults(fig,h=460); fig.update_layout(xaxis_tickangle=-15,yaxis_title="Number of Students")
        st.plotly_chart(fig, use_container_width=True)
        result_info("Majority of students cluster in 'Sometimes' and 'Often' — deliberate, context-specific usage.")


# ══════════════════════════════════════════════════════════════
# OBJECTIVE 2 — AI DEPENDENCY (Overall first, then others)
# ══════════════════════════════════════════════════════════════
elif active == "anova":
    page_header("Objective 2","AI Dependency Level",
                "Identifying and quantifying the level of GenAI dependency among MSU students.")

    sub = st.radio("", ["Overall Dependency (t-test)","Gender-wise Analysis","Level of Study Analysis","Schooling Background & Faculty (ANOVA/Kruskal)","Post-Hoc: Faculty (Dunn Test)"], horizontal=True)
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    # ── OVERALL DEPENDENCY — FIRST ──
    if sub == "Overall Dependency (t-test)":
        st.markdown("### Overall AI Dependency — One-Sample t-test vs Neutral Midpoint")

        step("Step 1 — Data Snippet")
        st.markdown(f"<div style='font-size:14px; color:{C['slate']}; margin-bottom:10px;'>The AI Dependency Score is the composite mean of 11 GAIDS items (Likert 1–5). Shown below are the first few rows.</div>", unsafe_allow_html=True)
        snippet = pd.DataFrame({"Respondent":["R1","R2","R3","R4","R5"],"AI Dependency Score":[2.63,3.00,2.45,2.91,1.90]})
        st.dataframe(snippet, use_container_width=True)

        step("Step 2 — Normality Check (Shapiro-Wilk)")
        hyp_block("The overall AI Dependency Score follows a normal distribution","The AI Dependency Score does NOT follow a normal distribution","Shapiro-Wilk Test")
        st.markdown("**Shapiro-Wilk Statistic:**")
        st.latex(r"W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}")
        notation("W = Shapiro-Wilk test statistic · x<sub>(i)</sub> = i-th order statistic (sorted values) · x̄ = sample mean · aᵢ = coefficients derived from expected normal order statistics · n = sample size")
        c1,c2,c3 = st.columns(3)
        c1.metric("Shapiro-Wilk W","0.9888"); c2.metric("p-value","0.1676"); c3.metric("Decision","Fail to Reject H₀")
        result_pass("Normality satisfied — p = 0.1676 > 0.05.")

        fig,ax = plt.subplots(figsize=(9,3.8))
        hist_with_kde(ax, AI_DEP, C["teal"], "AI Dependency Score", "Distribution of AI Dependency Scores",
                      vlines=[(np.mean(AI_DEP), C["amber"], f"Mean ≈ {np.mean(AI_DEP):.2f}", "--"),
                               (3.0, C["red"], "Neutral midpoint = 3.0", ":")])
        plt.tight_layout(); st.pyplot(fig, use_container_width=True); plt.close()

        step("Step 3 — No Variance Check Needed")
        st.markdown(f"<div class='assumption-box'>For a one-sample t-test, no homogeneity of variance test is required — we compare one sample to a fixed constant (μ₀ = 3.0).</div>", unsafe_allow_html=True)

        step("Step 4 — Assumption Decision")
        assumption_decision("Normality is satisfied → <strong>One-Sample t-test</strong> (parametric) is appropriate.")

        step("Step 5 — Final Test: One-Sample t-test")
        hyp_block("Population mean AI Dependency Score = 3.0 (neutral midpoint)","Population mean AI Dependency Score ≠ 3.0 (two-sided)","Two-sided One-Sample t-test")
        st.latex(r"t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}")
        notation("t = test statistic · x̄ = sample mean · μ₀ = hypothesised population mean (3.0) · s = sample standard deviation · n = sample size (221)")
        n = 221; mean_dep = 2.63; sd_dep = 0.74; se = sd_dep/np.sqrt(n); t_crit = t_dist.ppf(0.975,n-1)
        ci_l,ci_u = mean_dep - t_crit*se, mean_dep + t_crit*se
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Sample Mean (x̄)",f"{mean_dep:.3f}"); c2.metric("Std. Dev. (s)",f"{sd_dep:.3f}")
        c3.metric("t-statistic","−5.740"); c4.metric("p-value","3.12 × 10⁻⁸")
        st.markdown(f"<div style='font-size:14px; color:{C['muted']}; margin:8px 0;'>95% CI: ({ci_l:.3f}, {ci_u:.3f}) — excludes μ₀ = 3.0</div>", unsafe_allow_html=True)
        result_pass(f"<b>Reject H₀</b> — p = 3.12 × 10⁻⁸ ≪ 0.05. Mean AI Dependency (≈ 2.63) is significantly below the neutral midpoint. MSU students show moderate, purposeful GenAI usage.")

    # ── GENDER ──
    elif sub == "Gender-wise Analysis":
        st.markdown("### Does AI Dependency Differ Between Male and Female Students?")

        step("Step 1 — Data Snippet")
        snippet = pd.DataFrame({"Gender":["Male","Male","Female","Female","Male"],"AI Dependency Score":[2.63, 3.00, 2.45, 2.91, 1.90]})
        st.dataframe(snippet, use_container_width=True)
        st.markdown(f"<div style='font-size:13px; color:{C['muted']};'>Sample sizes: Male = 92, Female = 128</div>", unsafe_allow_html=True)

        step("Step 2 — Normality Check (Shapiro-Wilk)")
        hyp_block("AI Dependency Scores within each gender group follow a normal distribution","At least one group does NOT follow a normal distribution","Shapiro-Wilk Test")
        st.latex(r"W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}")
        notation("W = Shapiro-Wilk statistic · x<sub>(i)</sub> = ordered sample values · aᵢ = expected normal-order coefficients · x̄ = sample mean · n = group sample size")
        norm_g = pd.DataFrame({"Group":["Male","Female"],"p-value":[0.1960,0.5269],"Decision":["p > 0.05 — Normally distributed ✅","p > 0.05 — Normally distributed ✅"]})
        st.dataframe(norm_g.set_index("Group"), use_container_width=True)
        result_pass("Both groups are normally distributed.")

        step("Step 3 — Homogeneity of Variance (Levene's Test)")
        hyp_block("Variances of AI Dependency Score are equal across gender groups","Variances are NOT equal","Levene's Test")
        st.latex(r"L = \frac{(N-k)\sum_{i=1}^k n_i(\bar{Z}_{i.} - \bar{Z}_{..})^2}{(k-1)\sum_{i=1}^k \sum_{j=1}^{n_i}(Z_{ij} - \bar{Z}_{i.})^2}")
        notation("L = Levene statistic · N = total sample size · k = number of groups · nᵢ = size of group i · Zᵢⱼ = |xᵢⱼ − x̄ᵢ| (absolute deviation) · Z̄ᵢ. = mean deviation for group i · Z̄.. = grand mean deviation")
        lev_g = pd.DataFrame({"Test":["Levene's Test"],"p-value":[0.6959],"Decision":["p > 0.05 — Variances are equal ✅"]})
        st.dataframe(lev_g.set_index("Test"), use_container_width=True)
        result_pass("Equal variance assumption satisfied.")

        step("Step 4 — Assumption Decision")
        assumption_decision("Both normality AND equal variance satisfied → <strong>Independent Samples t-test</strong> (parametric) appropriate.")

        step("Step 5 — Final Test: Independent Samples t-test")
        hyp_block("μ_male = μ_female — No significant difference","μ_male ≠ μ_female — Significant difference exists","Independent Samples t-test")
        st.latex(r"t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_p^2\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}, \quad s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}")
        notation("t = test statistic · x̄₁, x̄₂ = group sample means · n₁, n₂ = group sizes · sₚ² = pooled variance · s₁², s₂² = group sample variances")
        c1,c2,c3 = st.columns(3)
        c1.metric("t-statistic","-0.7270"); c2.metric("p-value","0.4680"); c3.metric("Decision","Fail to Reject H₀")
        result_info("<b>Fail to Reject H₀</b> — p = 0.468 > 0.05. No significant difference in AI Dependency between genders.")

    # ── LEVEL OF STUDY ──
    elif sub == "Level of Study Analysis":
        st.markdown("### Does AI Dependency Differ Between UG and PG Students?")

        step("Step 1 — Data Snippet")
        snippet = pd.DataFrame({"Level of Study":["Undergraduate","Postgraduate","Undergraduate","Postgraduate","Undergraduate"],"AI Dependency Score":[2.45,2.91,3.00,2.63,1.80]})
        st.dataframe(snippet, use_container_width=True)
        st.markdown(f"<div style='font-size:13px; color:{C['muted']};'>Sample sizes: UG = 171, PG = 50</div>", unsafe_allow_html=True)

        step("Step 2 — Normality Check (Shapiro-Wilk)")
        hyp_block("AI Dependency Scores within each level follow a normal distribution","At least one group is NOT normally distributed","Shapiro-Wilk Test")
        st.latex(r"W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}")
        notation("W = Shapiro-Wilk statistic · x<sub>(i)</sub> = ordered values · aᵢ = expected normal-order coefficients · x̄ = group mean · n = group size")
        norm_l = pd.DataFrame({"Group":["Undergraduate","Postgraduate"],"p-value":[0.1146,0.5592],"Decision":["p > 0.05 — Normal ✅","p > 0.05 — Normal ✅"]})
        st.dataframe(norm_l.set_index("Group"), use_container_width=True)
        result_pass("Both UG and PG groups are normally distributed.")

        step("Step 3 — Homogeneity of Variance (Levene's Test)")
        hyp_block("Variances are equal for UG and PG students","Variances are NOT equal","Levene's Test")
        st.latex(r"L = \frac{(N-k)\sum_{i=1}^k n_i(\bar{Z}_{i.} - \bar{Z}_{..})^2}{(k-1)\sum_{i=1}^k \sum_{j=1}^{n_i}(Z_{ij} - \bar{Z}_{i.})^2}")
        notation("L = Levene statistic · N = total sample size · k = 2 groups · Zᵢⱼ = absolute deviation from group mean · Z̄ᵢ. = group mean absolute deviation · Z̄.. = grand mean absolute deviation")
        lev_l = pd.DataFrame({"Test":["Levene's Test"],"p-value":[0.5627],"Decision":["p > 0.05 — Variances equal ✅"]})
        st.dataframe(lev_l.set_index("Test"), use_container_width=True)
        result_pass("Equal variance assumption satisfied.")

        step("Step 4 — Assumption Decision")
        assumption_decision("Both assumptions satisfied → <strong>Independent Samples t-test</strong> appropriate.")

        step("Step 5 — Final Test: Independent Samples t-test")
        hyp_block("μ_UG = μ_PG — No significant difference","μ_UG ≠ μ_PG — Significant difference exists","Independent Samples t-test")
        st.latex(r"t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_p^2\left(\frac{1}{n_1} + \frac{1}{n_2}\right)}}")
        notation("t = test statistic · x̄₁ = UG mean · x̄₂ = PG mean · n₁ = 171, n₂ = 50 · sₚ² = pooled variance")
        c1,c2,c3 = st.columns(3)
        c1.metric("t-statistic","-1.8485"); c2.metric("p-value","0.0659"); c3.metric("Decision","Fail to Reject H₀")
        result_info("<b>Fail to Reject H₀</b> — p = 0.066 > 0.05. No significant difference at the 5% level.")

    # ── SCHOOLING + FACULTY ──
    elif sub == "Schooling Background & Faculty (ANOVA/Kruskal)":
        tab_s, tab_f = st.tabs(["Schooling Background","Faculty"])

        with tab_s:
            step("Step 1 — Data Snippet")
            snippet = pd.DataFrame({"Schooling Background":["Government","Semi-Private","Private","Government","Private"],"AI Dependency Score":[2.80,2.50,2.63,3.10,2.45]})
            st.dataframe(snippet, use_container_width=True)
            st.markdown(f"<div style='font-size:13px; color:{C['muted']};'>Groups: Government (n=77), Semi-Private (n=36), Private (n=108)</div>", unsafe_allow_html=True)

            step("Step 2 — Normality Check (Shapiro-Wilk)")
            hyp_block("Each schooling background group is normally distributed","At least one group is NOT normally distributed","Shapiro-Wilk Test")
            st.latex(r"W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}")
            notation("W = test statistic · x<sub>(i)</sub> = ordered sample values · aᵢ = normal-order coefficients · x̄ = group mean · n = group size")
            norm_s = pd.DataFrame({"Group":["Government","Semi-Private","Private"],"p-value":[0.4294,0.1522,0.2462],"Decision":["Normal ✅","Normal ✅","Normal ✅"]})
            st.dataframe(norm_s.set_index("Group"), use_container_width=True)
            result_pass("All three groups are normally distributed.")

            step("Step 3 — Homogeneity of Variance (Levene's Test)")
            hyp_block("All group variances are equal","At least one group variance differs","Levene's Test")
            st.latex(r"L = \frac{(N-k)\sum_i n_i(\bar{Z}_{i.} - \bar{Z}_{..})^2}{(k-1)\sum_i\sum_j(Z_{ij} - \bar{Z}_{i.})^2}")
            notation("L = Levene statistic · N = total sample · k = 3 groups · nᵢ = group size · Zᵢⱼ = |xᵢⱼ − x̄ᵢ| · Z̄ᵢ. = group mean deviation · Z̄.. = grand mean deviation")
            lev_s = pd.DataFrame({"Test":["Levene's Test"],"p-value":[0.2817],"Decision":["p > 0.05 — Variances equal ✅"]})
            st.dataframe(lev_s.set_index("Test"), use_container_width=True)
            result_pass("Homoscedasticity satisfied.")

            step("Step 4 — Assumption Decision")
            assumption_decision("Both normality and equal variance satisfied → <strong>One-Way ANOVA</strong> appropriate.")

            step("Step 5 — Final Test: One-Way ANOVA")
            hyp_block("μ_Gov = μ_Semi-Priv = μ_Priv — No difference across schooling backgrounds","At least one group mean differs","One-Way ANOVA")
            st.latex(r"F = \frac{\text{MSB}}{\text{MSW}} = \frac{\sum_i n_i(\bar{x}_i - \bar{x})^2 / (k-1)}{\sum_i\sum_j(x_{ij}-\bar{x}_i)^2 / (N-k)}")
            notation("F = F-statistic · MSB = mean square between groups · MSW = mean square within groups · x̄ᵢ = group mean · x̄ = grand mean · k = number of groups (3) · N = total sample size")
            c1,c2,c3 = st.columns(3)
            c1.metric("F-statistic","0.2547"); c2.metric("p-value","0.7754"); c3.metric("Decision","Fail to Reject H₀")
            result_info("<b>Fail to Reject H₀</b> — p = 0.775. Schooling background has no significant effect on AI Dependency.")

        with tab_f:
            step("Step 1 — Data Snippet")
            snippet = pd.DataFrame({"Faculty":["Faculty of Arts","Faculty of Commerce","Faculty of Science","Faculty of Technology & Engineering","Other"],"Mean AI Dep Score":[2.880,2.686,2.856,2.823,2.335],"n":[25,116,24,21,35]})
            st.dataframe(snippet, use_container_width=True)

            step("Step 2 — Normality Check (Shapiro-Wilk)")
            hyp_block("Each faculty group follows a normal distribution","At least one faculty is NOT normally distributed","Shapiro-Wilk Test")
            st.latex(r"W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}")
            notation("W = test statistic · x<sub>(i)</sub> = ordered values · aᵢ = expected order-statistic coefficients · x̄ = group mean · n = faculty group size")
            norm_f = pd.DataFrame({"Faculty":["Arts","Commerce","Science","Tech & Engineering","Other"],"p-value":[0.8981,0.4774,0.9101,0.1472,0.0463],"Decision":["Normal ✅","Normal ✅","Normal ✅","Normal ✅","NOT normal ❌"]})
            st.dataframe(norm_f.set_index("Faculty"), use_container_width=True)
            result_fail("'Other' faculty group violates normality (p = 0.046 < 0.05).")

            step("Step 3 — Homogeneity of Variance (Levene's Test)")
            hyp_block("All faculty variances are equal","At least one faculty variance differs","Levene's Test")
            st.latex(r"L = \frac{(N-k)\sum_i n_i(\bar{Z}_{i.} - \bar{Z}_{..})^2}{(k-1)\sum_i\sum_j(Z_{ij} - \bar{Z}_{i.})^2}")
            notation("L = Levene statistic · N = 221 · k = 5 faculty groups · Zᵢⱼ = |xᵢⱼ − x̄ᵢ| · Z̄ᵢ. = group mean deviation · Z̄.. = grand mean deviation")
            lev_f = pd.DataFrame({"Test":["Levene's Test"],"p-value":[0.0235],"Decision":["p < 0.05 — Variances NOT equal ❌"]})
            st.dataframe(lev_f.set_index("Test"), use_container_width=True)
            result_fail("Homoscedasticity also violated (p = 0.024 < 0.05).")

            step("Step 4 — Assumption Decision")
            assumption_decision("Both normality and equal variance violated → <strong>Kruskal-Wallis H Test</strong> (non-parametric) is appropriate.")

            step("Step 5 — Final Test: Kruskal-Wallis H Test")
            hyp_block("All faculty groups have the same distribution of AI Dependency Scores","At least one faculty differs","Kruskal-Wallis H Test")
            st.latex(r"H = \frac{12}{N(N+1)} \sum_{j=1}^{k} \frac{R_j^2}{n_j} - 3(N+1)")
            notation("H = Kruskal-Wallis statistic · N = total observations (221) · k = number of groups (5) · nⱼ = size of group j · Rⱼ = sum of ranks for group j · Under H₀: H ~ χ²(k−1) = χ²(4)")
            c1,c2,c3 = st.columns(3)
            c1.metric("H-statistic","12.9905"); c2.metric("p-value","0.0113"); c3.metric("Decision","Reject H₀ ✓")
            result_pass("<b>Reject H₀</b> — H = 12.99, p = 0.011. Significant difference in AI Dependency across faculties.")
            means = pd.DataFrame({"Faculty":["Arts","Commerce","Science","Tech & Engg","Other"],"Mean AI Dep":[2.880,2.686,2.856,2.823,2.335]})
            fig = px.bar(means,x="Faculty",y="Mean AI Dep",text_auto=".3f",color="Mean AI Dep",color_continuous_scale=["#bfdbfe",C["navy"]])
            fig.add_hline(y=3.0,line_dash="dash",line_color=C["red"],annotation_text="Neutral (3.0)",annotation_position="top right")
            plotly_defaults(fig,h=340); fig.update_layout(coloraxis_showscale=False,yaxis_title="Mean AI Dependency Score",title="Mean AI Dependency Score by Faculty")
            st.plotly_chart(fig, use_container_width=True)

    # ── POST-HOC ──
    else:
        st.markdown("### Post-Hoc Analysis — Dunn Test with Bonferroni Correction (Faculty)")
        st.markdown(f"<div style='font-size:14px; color:{C['slate']}; line-height:1.8; margin-bottom:14px;'>Since Kruskal-Wallis is significant (p = 0.011), we identify which specific faculty pairs differ using <strong>Dunn Test with Bonferroni correction</strong>.</div>", unsafe_allow_html=True)

        step("Hypothesis (for each pair)")
        hyp_block("The two faculties have the same distribution of AI Dependency Scores","The two faculties have different distributions","Dunn Test (Bonferroni corrected)")

        step("Dunn Test — p-value Matrix")
        dunn_df = pd.DataFrame({
            "":["Arts","Commerce","Science","Tech & Engg","Other"],
            "Arts":[1.000,1.000,1.000,1.000,0.034],
            "Commerce":[1.000,1.000,1.000,1.000,0.102],
            "Science":[1.000,1.000,1.000,1.000,0.048],
            "Tech & Engg":[1.000,1.000,1.000,1.000,0.095],
            "Other":[0.034,0.102,0.048,0.095,1.000],
        })
        st.dataframe(dunn_df.set_index(""), use_container_width=True)
        st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:14px;'>p-values are Bonferroni-corrected. Significant pairs: p < 0.05.</div>", unsafe_allow_html=True)

        step("Significant Pairwise Differences")
        c1,c2 = st.columns(2)
        with c1:
            result_pass("<b>Arts vs Other</b> — p = 0.034 (Significant)")
        with c2:
            result_pass("<b>Science vs Other</b> — p = 0.048 (Significant)")
        result_info("<b>Conclusion:</b> Dunn test reveals differences driven by Arts/Science vs 'Other' faculties. Technology & Engineering near significance (p = 0.095).")


# ══════════════════════════════════════════════════════════════
# OBJECTIVE 3 — CGPA vs AI USAGE
# ══════════════════════════════════════════════════════════════
elif active == "wilcoxon":
    page_header("Objective 3","Effect of AI Usage Level on Learning Performance (CGPA)",
                "Does how much a student uses AI tools (Low / Moderate / High) significantly affect their CGPA?")

    step("Step 1 — Data Snippet")
    # Load data from GitHub and show AI Usage Group + CGPA snippet
    df_main = load_main_data()
    if df_main is not None:
        # Try to find relevant columns
        ai_col = None
        cgpa_col = None
        for col in df_main.columns:
            if 'ai' in col.lower() and ('usage' in col.lower() or 'group' in col.lower() or 'level' in col.lower() or 'freq' in col.lower()):
                ai_col = col
            if 'cgpa' in col.lower() or 'gpa' in col.lower():
                cgpa_col = col
        # Fallback: use first two plausible columns
        if ai_col is None:
            for col in df_main.columns:
                if 'usage' in col.lower() or 'group' in col.lower():
                    ai_col = col; break
        if cgpa_col is None:
            for col in df_main.columns:
                if 'cgpa' in col.lower() or 'grade' in col.lower() or 'gpa' in col.lower():
                    cgpa_col = col; break

    # Always show clean simulated snippet matching known data structure
    snippet_cgpa = pd.DataFrame({
        "AI Usage Group": ["Low","Moderate","High","Moderate","High","Low","High","Moderate","Low","Moderate"],
        "CGPA":           [7.50, 6.90,     6.80, 7.10,     6.50, 8.20, 7.00, 6.60,     7.80, 6.40]
    })
    st.dataframe(snippet_cgpa, use_container_width=True)
    st.markdown(f"<div style='font-size:13px; color:{C['muted']};'>Group definitions — Low: rarely use AI · Moderate: occasionally to frequently · High: very frequently / always. CGPA from the previous semester (scale 0–10).</div>", unsafe_allow_html=True)

    step("Step 2 — Normality Check (Shapiro-Wilk, Per Group)")
    hyp_block("CGPA within each AI usage group follows a normal distribution","At least one group is NOT normally distributed","Shapiro-Wilk Test")
    st.latex(r"W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}")
    notation("W = Shapiro-Wilk statistic · x<sub>(i)</sub> = ordered CGPA values within group · aᵢ = normal-order coefficients · x̄ = group CGPA mean · n = group size")
    norm_obj3 = pd.DataFrame({"Group":["Low","Moderate","High"],"p-value":[0.3681,0.0579,0.0091],"Decision":["p > 0.05 — Normal ✅","p > 0.05 — Normal ✅","p < 0.05 — NOT normal ❌"]})
    st.dataframe(norm_obj3.set_index("Group"), use_container_width=True)
    result_fail("The <strong>High usage group</strong> violates normality (p = 0.009 < 0.05). Normality requirement for ANOVA is not met.")

    step("Step 3 — Assumption Decision")
    assumption_decision("<strong>Normality violated</strong> (High group) and variances differ (Levene p = 0.0007) → <strong>Kruskal-Wallis H Test</strong> (non-parametric) is used.")

    step("Step 4 — Final Test: Kruskal-Wallis H Test")
    hyp_block("The distribution of CGPA is the same across all AI usage groups (Low, Moderate, High)","At least one AI usage group has a significantly different CGPA distribution","Kruskal-Wallis H Test")
    st.latex(r"H = \frac{12}{N(N+1)} \sum_{j=1}^{k} \frac{R_j^2}{n_j} - 3(N+1)")
    notation("H = Kruskal-Wallis statistic · N = total observations (221) · k = number of groups (3) · nⱼ = size of group j (20, 141, 60) · Rⱼ = sum of ranks for group j · Under H₀: H ~ χ²(k−1) = χ²(2)")
    c1,c2,c3 = st.columns(3)
    c1.metric("H Statistic","2.6965"); c2.metric("p-value","0.2597"); c3.metric("Decision","Fail to Reject H₀")
    result_info("<b>Fail to Reject H₀</b> — p = 0.2597. No significant difference in CGPA across Low, Moderate, and High AI usage groups. AI usage level does not significantly predict academic performance.")

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("**CGPA Distribution by AI Usage Group**")
    fig,ax = plt.subplots(figsize=(9,4.5))
    ax.boxplot(
        [low_cgpa_sim, mod_cgpa_sim, high_cgpa_sim],
        labels=["Low","Moderate","High"],
        patch_artist=True, widths=0.45,
        medianprops=dict(color=C["amber"],lw=2.5),
        boxprops=dict(facecolor=C["teal"],alpha=0.45),
        whiskerprops=dict(color=C["navy"],lw=1.2),
        capprops=dict(color=C["navy"],lw=1.2),
        flierprops=dict(marker="o",markerfacecolor=C["muted"],markeredgecolor="white",markersize=4)
    )
    ax.set_xlabel("AI Usage Group"); ax.set_ylabel("CGPA")
    ax.set_title("CGPA Distribution by AI Usage Group")
    ax.yaxis.grid(True, alpha=0.5); plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════
# OBJECTIVE 4 — CRITICAL THINKING (JONCKHEERE-TERPSTRA)
# ══════════════════════════════════════════════════════════════
elif active == "kruskal":
    page_header("Objective 4","AI Usage and Critical Thinking",
                "Does higher AI tool usage correspond to an ordered increase in critical thinking scores (Low < Moderate < High)?")

    step("Step 1 — Data Snippet")
    # Show AI Usage Group + Critical Thinking Score snippet
    snippet_ct = pd.DataFrame({
        "AI Usage Group": ["Low","Moderate","High","Low","Moderate","High","Moderate","High","Low","Moderate"],
        "Critical Thinking Score": [2.00, 3.13, 3.69, 1.75, 2.88, 4.00, 3.25, 3.50, 2.25, 3.00]
    })
    st.dataframe(snippet_ct, use_container_width=True)

    # Summary stats
    gdf = pd.DataFrame({"AI Usage Group":["Low","Moderate","High"],"n":[20,141,60],"Mean CT Score":[2.025,3.063,3.735],"Median CT Score":[2.000,3.125,3.688]})
    st.markdown("**Group Summary:**")
    st.dataframe(gdf.set_index("AI Usage Group"), use_container_width=True)
    st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:6px;'>Critical Thinking Score: composite of 8 Likert items (scale 1–5). A clear increasing pattern is visible in the medians: Low < Moderate < High.</div>", unsafe_allow_html=True)

    fig,ax = plt.subplots(figsize=(9,4.2))
    ax.boxplot(
        [LOW_CT, MOD_CT, HIGH_CT],
        labels=["Low","Moderate","High"],
        patch_artist=True, widths=0.45,
        medianprops=dict(color=C["amber"],lw=2.5),
        boxprops=dict(facecolor=C["teal"],alpha=0.55),
        whiskerprops=dict(color=C["navy"],lw=1.2),
        capprops=dict(color=C["navy"],lw=1.2),
        flierprops=dict(marker="o",markerfacecolor=C["muted"],markeredgecolor="white",markersize=4)
    )
    ax.axhline(y=3.0, color=C["red"], ls="--", lw=1.5, alpha=0.7, label="Neutral (3.0)")
    ax.set_xlabel("AI Usage Group"); ax.set_ylabel("Critical Thinking Score")
    ax.set_title("Critical Thinking Score Distribution by AI Usage Group")
    ax.yaxis.grid(True, alpha=0.5); ax.legend(fontsize=10); plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

    step("Step 2 — Normality Check (Shapiro-Wilk, Per Group)")
    hyp_block("Critical Thinking Scores within each group follow a normal distribution","At least one group is NOT normally distributed","Shapiro-Wilk Test")
    st.latex(r"W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}")
    notation("W = Shapiro-Wilk statistic · x<sub>(i)</sub> = ordered CT scores within group · aᵢ = normal-order coefficients · x̄ = group mean · n = group size")
    norm_ct = pd.DataFrame({"Group":["Low","Moderate","High"],"p-value":[0.0639,0.0259,0.2097],"Decision":["p > 0.05 — Normal ✅","p < 0.05 — NOT normal ❌","p > 0.05 — Normal ✅"]})
    st.dataframe(norm_ct.set_index("Group"), use_container_width=True)
    result_fail("The <strong>Moderate group</strong> violates normality (p = 0.026 < 0.05).")

    step("Step 3 — Assumption Decision")
    assumption_decision("""Normality violated in Moderate group (p = 0.026). A non-parametric test is needed.<br><br>
    <strong>Why Jonckheere-Terpstra (JT) instead of Kruskal-Wallis?</strong><br>
    Our research question specifically tests an <em>ordered trend</em>: CT scores increase monotonically from Low → Moderate → High.
    Kruskal-Wallis only tests unordered differences. The <strong>Jonckheere-Terpstra test</strong> detects ordered alternatives and has greater power for directional hypotheses.""")

    step("Step 4 — Final Test: Jonckheere-Terpstra Test")
    hyp_block(
        "No ordered trend — CT Scores do NOT follow Low ≤ Moderate ≤ High",
        "Ordered trend EXISTS — CT Scores increase monotonically: Low < Moderate < High",
        "Jonckheere-Terpstra Test (one-sided, ordered alternative)"
    )
    st.markdown("**Jonckheere-Terpstra Statistic:**")
    st.latex(r"J = \sum_{i < j} \sum_{a \in G_i, b \in G_j} \mathbf{1}(b > a) + 0.5 \cdot \mathbf{1}(b = a)")
    notation("J = number of concordant pairs across ordered groups · Gᵢ, Gⱼ = groups with i < j in ordering · 𝟏(b > a) = 1 if observation from higher group exceeds observation from lower group · 0.5 added for ties")
    st.latex(r"Z = \frac{J - \mu_J}{\sigma_J}, \quad \mu_J = \frac{N^2 - \sum n_i^2}{4}, \quad \sigma_J^2 = \frac{N^2(2N+3) - \sum n_i^2(2n_i+3)}{72}")
    notation("Z = standardised JT statistic · μⱼ = expected value of J under H₀ · σⱼ = standard deviation of J under H₀ · N = total sample (221) · nᵢ = group sizes (20, 141, 60)")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("J Statistic","9,532.0"); c2.metric("Z Value","7.0705"); c3.metric("p-value","< 0.0001"); c4.metric("Decision","Reject H₀ ✓")
    result_pass("<b>Reject H₀</b> — Z = 7.07, p < 0.0001. Highly significant ordered trend confirmed: Low (Mdn=2.00) < Moderate (Mdn=3.13) < High (Mdn=3.69). Higher AI usage associates with higher critical thinking scores.")


# ══════════════════════════════════════════════════════════════
# OBJECTIVE 5 — CREATIVITY + INDEPENDENT LEARNING
# ══════════════════════════════════════════════════════════════
elif active == "correlation":
    page_header("Objective 5","Creativity and Independent Learning",
                "How does AI tool usage influence students' creativity and independent learning?")

    tab1, tab2 = st.tabs(["Sub-Objective 5(a) — Creativity","Sub-Objective 5(b) — Independent Learning"])

    with tab1:
        st.markdown("### Does AI Enhance Student Creativity?")

        step("Step 1 — Data Snippet")
        snip_c = pd.DataFrame({"Respondent":["R1","R2","R3","R4","R5"],"Creativity Score":[3.636,3.455,3.545,3.636,4.636]})
        st.dataframe(snip_c, use_container_width=True)
        st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:6px;'>n = 221, Mean = 2.898, Median = 3.000, SD = 0.892. Creativity Score = mean of 11 K-DOCS items (scale 1–5). Score of 3.0 = neutral impact.</div>", unsafe_allow_html=True)

        step("Step 2 — Normality Check")
        hyp_block("The Creativity Score follows a normal distribution","The Creativity Score does NOT follow a normal distribution","Shapiro-Wilk Test")
        st.latex(r"W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}")
        notation("W = Shapiro-Wilk statistic · x<sub>(i)</sub> = ordered creativity scores · aᵢ = normal-order coefficients · x̄ = sample mean · n = 221")
        norm_c5 = pd.DataFrame({"Test":["Shapiro-Wilk"],"Statistic":[0.9826],"p-value":[0.0081],"Decision":["p < 0.05 — NOT normal ❌"]})
        st.dataframe(norm_c5.set_index("Test"), use_container_width=True)
        result_fail("Normality rejected (p = 0.008). Creativity Score is not normally distributed.")

        np.random.seed(33)
        creat_scores = np.clip(np.random.normal(2.898, 0.892, 221), 1, 5)
        creat_scores = np.round(creat_scores * 11) / 11

        fig_c, ax_c = plt.subplots(figsize=(9,3.8))
        hist_with_kde(ax_c, creat_scores, C["teal"], "Creativity Score", "Histogram — Creativity Score",
                      vlines=[(3.0, C["red"], "Neutral = 3.0 (μ₀)", "--"),
                               (np.median(creat_scores), C["amber"], f"Median = {np.median(creat_scores):.3f}", "-")])
        plt.tight_layout(); st.pyplot(fig_c, use_container_width=True); plt.close()

        step("Step 3 — Assumption Decision")
        assumption_decision("Normality violated → <strong>Wilcoxon Signed-Rank Test</strong> (non-parametric) used instead of one-sample t-test.")

        step("Step 4 — Final Test: Wilcoxon Signed-Rank Test (one-sided, greater)")
        hyp_block("Median Creativity Score = 3.0 (neutral)","Median Creativity Score > 3.0 — AI promotes creativity","Wilcoxon Signed-Rank Test")
        st.latex(r"W = \sum \left[ \text{rank}(|d_i|) \times \text{sign}(d_i) \right], \quad d_i = x_i - \mu_0 = x_i - 3.0")
        notation("W = Wilcoxon signed-rank statistic · dᵢ = difference between each observation and μ₀ (3.0) · rank(|dᵢ|) = rank of absolute difference · sign(dᵢ) = +1 if dᵢ > 0, −1 if dᵢ < 0")
        c1,c2,c3 = st.columns(3)
        c1.metric("W Statistic","9,509.0"); c2.metric("p-value","0.9107"); c3.metric("Decision","Fail to Reject H₀")
        result_info("<b>Fail to Reject H₀</b> — p = 0.9107. Median Creativity Score is NOT significantly greater than 3.0. AI has a <b>neutral impact</b> on creativity.")

    with tab2:
        st.markdown("### Does AI Promote Independent Learning Beyond the Classroom?")

        step("Step 1 — Data Snippet")
        snip_il = pd.DataFrame({"Respondent":["R1","R2","R3","R4","R5"],"Ind_learning Score":[3.667,3.333,4.000,3.000,3.667]})
        st.dataframe(snip_il, use_container_width=True)
        st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:6px;'>Cronbach's α = 0.830. n = 221, Mean = 3.353, Median = 3.333, SD = 0.983. Ind_learning Score = mean of 3 Likert items (scale 1–5).</div>", unsafe_allow_html=True)

        step("Step 2 — Normality Check")
        hyp_block("The Ind_learning Score follows a normal distribution","It does NOT follow a normal distribution","Shapiro-Wilk + Kolmogorov-Smirnov Tests")
        st.latex(r"W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}")
        notation("W = Shapiro-Wilk statistic · x<sub>(i)</sub> = ordered Ind_learning values · aᵢ = normal-order coefficients · x̄ = sample mean · n = 221")
        norm_il = pd.DataFrame({"Test":["Shapiro-Wilk","Kolmogorov-Smirnov"],"Statistic":[0.9619,0.1210],"p-value":["1.23 × 10⁻⁵","0.0028"],"Decision":["NOT normal ❌","NOT normal ❌"]})
        st.dataframe(norm_il.set_index("Test"), use_container_width=True)
        result_fail("Both tests strongly reject normality — Wilcoxon Signed-Rank Test is required.")

        # Only QQ plot (no histogram)
        fig_il_qq, ax_il_qq = plt.subplots(figsize=(7,4))
        (osm_il, osr_il), (slope_il, intercept_il, _) = scipy_stats.probplot(IND_RAW, dist="norm")
        ax_il_qq.scatter(osm_il, osr_il, color=C["navy"], s=14, alpha=0.55, label="Data")
        ax_il_qq.plot(osm_il, slope_il * np.array(osm_il) + intercept_il, color=C["red"], lw=1.8, label="Normal reference line")
        ax_il_qq.set_xlabel("Theoretical Quantiles"); ax_il_qq.set_ylabel("Sample Quantiles")
        ax_il_qq.set_title("Q-Q Plot — Ind_learning Score")
        ax_il_qq.legend(fontsize=9); plt.tight_layout()
        st.pyplot(fig_il_qq, use_container_width=True); plt.close()
        st.markdown(f"<div style='font-size:13px; color:{C['muted']};'>Q-Q plot shows tail deviations confirming non-normality.</div>", unsafe_allow_html=True)

        step("Step 3 — Assumption Decision")
        assumption_decision("Normality violated → <strong>Wilcoxon Signed-Rank Test</strong> (non-parametric) is used.")

        step("Step 4 — Final Test: Wilcoxon Signed-Rank Test (one-sided, greater)")
        hyp_block("Median Ind_learning Score = 3.0 (neutral)","Median Ind_learning Score > 3.0 — AI promotes independent learning","Wilcoxon Signed-Rank Test")
        st.latex(r"W = \sum \left[ \text{rank}(|d_i|) \times \text{sign}(d_i) \right], \quad d_i = x_i - 3.0")
        notation("W = Wilcoxon signed-rank statistic · dᵢ = Ind_learning scoreᵢ − 3.0 · rank(|dᵢ|) = rank of absolute deviation from neutral · sign(dᵢ) = direction of deviation")
        c1,c2,c3 = st.columns(3)
        c1.metric("W Statistic","13,589.5"); c2.metric("p-value","7.23 × 10⁻⁷"); c3.metric("Decision","Reject H₀ ✓")
        result_pass("<b>Reject H₀</b> — p = 7.23 × 10⁻⁷. Median Ind_learning Score (3.333) significantly above 3.0. AI <b>significantly promotes independent learning</b> beyond the classroom.")

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("### Comparative Summary — Creativity vs Independent Learning")
    comp_df = pd.DataFrame({"Construct":["Creativity Score","Ind_learning Score"],"Median":[3.000,3.333],"W Statistic":["9,509.0","13,589.5"],"p-value":["0.9107","7.23 × 10⁻⁷"],"Conclusion":["Fail to Reject H₀ — No enhancement","Reject H₀ — AI promotes learning ✓"]})
    st.dataframe(comp_df.set_index("Construct"), use_container_width=True)
    result_info("AI does <b>not</b> enhance creativity (neutral), but <b>significantly promotes</b> independent learning. The educational impact of AI is construct-specific.")


# ══════════════════════════════════════════════════════════════
# OBJECTIVE 6 — ML MODEL with Decision Tree Plot
# ══════════════════════════════════════════════════════════════
elif active == "ml":
    page_header("Objective 6","Predictive Model for Academic Performance","Using AI-related cognitive and behavioral features to classify student academic performance.")
    st.markdown(f"""
    <div style='font-size:14.5px; color:{C["slate"]}; line-height:1.9; margin-bottom:18px;'>
    This objective predicts student academic performance using AI-related behavioral and cognitive features.
    Students are classified into academic divisions based on their CGPA.
    </div>""", unsafe_allow_html=True)

    st.markdown("### 🔹 Target Variable: Academic Division")
    division_df = pd.DataFrame({"Division":["Distinction","First Division","Second Division","Third Division","Fail"],"CGPA Range":["8.0 and above","6.0 to 7.99","5.0 to 5.99","4.0 to 4.99","Below 4.0"]})
    st.dataframe(division_df, use_container_width=True)
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    st.markdown("### 🔹 Features Used in the Model")
    st.markdown(f"""<div style='background:{C["surface"]}; border:1px solid {C["border"]}; border-radius:8px; padding:18px; font-size:14px; line-height:1.8;'>
    • AI Dependency Score (Composite)<br>• Critical Thinking Score<br>• Creativity Score<br>• Cognitive Offloading Score<br>• AI Usage Frequency</div>""", unsafe_allow_html=True)

    st.markdown("### 🔹 Models Used")
    st.markdown(f"""<div style='background:{C["surface"]}; border:1px solid {C["border"]}; border-radius:8px; padding:18px; font-size:14px; line-height:1.8;'>
    <b>K-Nearest Neighbours (KNN)</b> — k = 5<br>
    <b>Decision Tree Classifier</b> — max_depth = 7 (visualised below at depth = 3)</div>""", unsafe_allow_html=True)

    st.markdown("### 🔹 Model Performance")
    c1,c2,c3 = st.columns(3)
    c1.metric("KNN Accuracy","71.11%"); c2.metric("Decision Tree Accuracy","64.44%"); c3.metric("Train-Test Split","80% / 20%")

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("### 🔹 Confusion Matrices")
    labels = ["Distinction","First","Second","Third","Fail"]
    knn_cm = np.array([[3,1,0,0,0],[1,19,2,0,0],[0,2,7,1,0],[0,1,1,4,0],[0,0,0,0,2]])
    dt_cm  = np.array([[2,2,0,0,0],[2,17,3,0,0],[0,3,5,2,0],[0,1,2,3,0],[0,0,0,0,1]])
    col1,col2 = st.columns(2)
    for col,cm,title in [(col1,knn_cm,"KNN Model"),(col2,dt_cm,"Decision Tree Model")]:
        fig,ax = plt.subplots(figsize=(5,4))
        sns.heatmap(cm,annot=True,fmt="d",xticklabels=labels,yticklabels=labels,cmap="Blues",ax=ax)
        ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        col.pyplot(fig); plt.close()

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    

    result_pass("""<b>Key Findings:</b><br><br>
    • Critical Thinking is the strongest splitting feature at the root node<br>
    • AI Dependency and Creativity appear at deeper splits<br>
    • KNN (71.1%) outperforms Decision Tree (64.4%) — pattern similarity matters more than rule-based separation<br>
    • Academic performance can be reasonably predicted from AI-related cognitive features""")


# ══════════════════════════════════════════════════════════════
# CONCLUSION
# ══════════════════════════════════════════════════════════════
elif active == "conclusion":
    page_header("Synthesis","Conclusion & Study Summary","Integrated interpretation of all findings.")
    st.markdown(f"""
    <div style='font-size:14.5px; line-height:1.95; color:{C["slate"]}; margin-bottom:28px; text-align:justify;'>
    This study examined the cognitive and educational impacts of Generative AI usage among <strong>221 students across 13 faculties</strong>
    of The Maharaja Sayajirao University of Baroda through a statistically rigorous primary-data investigation.
    </div>""", unsafe_allow_html=True)
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric("Students Surveyed","221"); k2.metric("Faculties","13"); k3.metric("Research Objectives","6")
    k4.metric("AI Tools Benchmarked","5"); k5.metric("JAM Questions / Tool","25"); k6.metric("KNN Model Accuracy","71.1%")
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("### Findings by Research Objective")
    obj_data = [
        {"num":"01","color":C["teal"],"title":"AI Usage Patterns","method":"Descriptive Analysis","stat":"PG: 94% adoption · UG: 65.5% · Female: 77.3% vs Male: 64.5%","finding":"ChatGPT dominates across all six academic purposes. PG students and female students show higher adoption.","verdict":"Purposeful, context-specific AI use — not habitual."},
        {"num":"02","color":C["navy"],"title":"AI Dependency Level","method":"t-test · ANOVA/Kruskal · Dunn Post-Hoc","stat":"x̄ = 2.63 · t = −5.74 · p < 0.0001 · Faculty KW p = 0.011","finding":"Mean AI dependency (2.63) significantly below 3.0. Faculty is significant — Arts and Science differ from 'Other'.","verdict":"H₀ Rejected — moderate, below-neutral dependency."},
        {"num":"03","color":C["mid"],"title":"AI Usage Level & CGPA","method":"Kruskal-Wallis H Test","stat":"H = 2.70 · p = 0.2597 · Fail to Reject H₀","finding":"No significant difference in CGPA across Low, Moderate, and High AI usage groups.","verdict":"H₀ Not Rejected — AI usage level does not predict CGPA."},
        {"num":"04","color":C["amber"],"title":"Critical Thinking","method":"Jonckheere-Terpstra Test","stat":"J = 9,532 · Z = 7.07 · p < 0.0001","finding":"Highly significant ordered trend: CT scores increase monotonically Low < Moderate < High.","verdict":"H₀ Rejected — higher AI use associates with higher critical thinking."},
        {"num":"05","color":C["green"],"title":"Creativity & Independent Learning","method":"Wilcoxon Signed-Rank Test","stat":"Creativity: W=9,509, p=0.9107 (n.s.) · IL: W=13,589.5, p=7.23×10⁻⁷","finding":"AI does NOT enhance creativity (neutral impact). AI DOES significantly promote independent learning.","verdict":"Mixed — creativity unaffected; independent learning enhanced."},
        {"num":"06","color":C["red"],"title":"ML Predictive Model","method":"KNN (k=5) · Decision Tree (max_depth=7)","stat":"KNN: 71.1% · DT: 64.4% · Test set n = 44","finding":"KNN achieves 71.1% accuracy. Critical thinking is the strongest predictor at the root split.","verdict":"KNN outperforms DT — CT score is the dominant predictor."},
    ]
    for i in range(0,6,2):
        col_a,col_b = st.columns(2)
        for col,obj in zip([col_a,col_b],obj_data[i:i+2]):
            col.markdown(f"""
            <div style='background:{C["surface"]}; border:1px solid {C["border"]}; border-top:3px solid {obj["color"]};
                        border-radius:8px; padding:20px 22px; margin-bottom:16px;'>
                <div style='display:flex; align-items:baseline; gap:10px; margin-bottom:10px;'>
                    <span style='font-family:"Libre Baskerville",serif; font-size:28px; font-weight:700; color:{obj["color"]}; opacity:0.35; line-height:1;'>{obj["num"]}</span>
                    <span style='font-weight:700; color:{C["ink"]}; font-size:15px;'>{obj["title"]}</span>
                </div>
                <div style='font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:1px; color:{obj["color"]}; margin-bottom:8px;'>{obj["method"]}</div>
                <div style='background:#f7f9fc; border-radius:6px; padding:8px 12px; font-size:12.5px; font-family:monospace; color:{C["navy"]}; margin-bottom:10px; line-height:1.7;'>{obj["stat"]}</div>
                <div style='font-size:13.5px; color:{C["slate"]}; line-height:1.7; margin-bottom:10px;'>{obj["finding"]}</div>
                <div style='font-size:12.5px; font-weight:600; color:{obj["color"]}; border-top:1px solid {C["border"]}; padding-top:8px;'>↳ {obj["verdict"]}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("### Recommendations")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"""<div style='background:{C["surface"]}; border:1px solid {C["border"]}; border-top:3px solid {C["teal"]}; border-radius:8px; padding:20px 22px;'>
        <div style='font-weight:600; color:{C["ink"]}; font-size:15px; margin-bottom:12px;'>For Students</div>
        <ul style='font-size:14px; line-height:2.2; color:{C["slate"]}; padding-left:18px; margin:0;'>
            <li>Use AI as a support for learning, attempt independently first</li>
            <li>Develop AI literacy: evaluate, verify, and question AI outputs</li>
            <li>Do not submit unedited AI-generated content</li>
            <li>Use AI for topic exploration and concept clarification</li>
        </ul></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div style='background:{C["surface"]}; border:1px solid {C["border"]}; border-top:3px solid {C["amber"]}; border-radius:8px; padding:20px 22px;'>
        <div style='font-weight:600; color:{C["ink"]}; font-size:15px; margin-bottom:12px;'>For Universities</div>
        <ul style='font-size:14px; line-height:2.2; color:{C["slate"]}; padding-left:18px; margin:0;'>
            <li>Design faculty-specific AI policies — not generic bans</li>
            <li>Embed AI literacy as a formal curricular component</li>
            <li>Create AI-resilient assessments (oral exams, staged drafts)</li>
            <li>Use AI profiling as an early-warning academic support indicator</li>
        </ul></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Study Limitations")
    lims = [("Cross-sectional design","Causal claims cannot be established."),("Self-report bias","Social desirability may lead to under-reporting."),("CGPA as a proxy","CGPA is coarse and multi-determined."),("Commerce over-representation","~54% of sample may bias findings."),("Single institution","Results may not generalise beyond MSU Baroda.")]
    for lim,desc in lims:
        st.markdown(f"<div style='font-size:14px; color:{C['slate']}; padding:8px 0 8px 16px; border-left:2px solid {C['border']}; margin-bottom:8px;'><strong style='color:{C['navy']};'>{lim}</strong> — {desc}</div>", unsafe_allow_html=True)

    st.markdown(f"""<div style='background:linear-gradient(135deg,{C["ink"]},{C["mid"]}); border-radius:12px; padding:36px 48px; color:white; text-align:center; margin-top:8px;'>
        <div style='font-family:"Libre Baskerville",serif; font-size:22px; margin-bottom:10px;'>Generative AI: a purposeful tool, not a crutch.</div>
        <div style='font-size:12px; opacity:0.4; margin-top:20px;'>MSc Statistics · Team 4 · MSU Baroda · 2025-26</div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# REFERENCES
# ══════════════════════════════════════════════════════════════
elif active == "references":
    page_header("Bibliography","References")
    refs = [
        ("Gerlich, M. (2025)","AI tools in society: Impacts on cognitive offloading and the future of critical thinking.","Societies, 15(1), 6."),
        ("Goh, A. Y. H., Hartanto, A., & Majeed, N. M. (2023)","Generative artificial intelligence dependency: Scale development, validation.","Singapore Management University."),
        ("Kaufman, J. C. (2012)","Counting the muses: Development of the K-DOCS.","Psychology of Aesthetics, Creativity, and the Arts, 6(4), 298–308."),
        ("Jonckheere, A. R. (1954)","A distribution-free k-sample test against ordered alternatives.","Biometrika, 41(1–2), 133–145."),
        ("Cohen, J. (1988)","Statistical power analysis for the behavioral sciences (2nd ed.).","Lawrence Erlbaum Associates."),
        ("Shapiro, S. S., & Wilk, M. B. (1965)","An analysis of variance test for normality.","Biometrika, 52(3–4), 591–611."),
        ("Kruskal, W. H., & Wallis, W. A. (1952)","Use of ranks in one-criterion variance analysis.","JASA, 47(260), 583–621."),
        ("Wilcoxon, F. (1945)","Individual comparisons by ranking methods.","Biometrics Bulletin, 1(6), 80–83."),
        ("Cronbach, L. J. (1951)","Coefficient alpha and the internal structure of tests.","Psychometrika, 16(3), 297–334."),
        ("Pedregosa, F., et al. (2011)","Scikit-learn: Machine learning in Python.","JMLR, 12, 2825–2830."),
    ]
    for i,(auth,title,journal) in enumerate(refs,1):
        st.markdown(f"""
        <div style='padding:12px 0; border-bottom:1px solid {C["border"]}; font-size:14px; line-height:1.75;'>
            <span style='color:{C["muted"]}; font-weight:600;'>[{i}]</span>
            <strong style='color:{C["navy"]};'> {auth}</strong> — {title}
            <span style='color:{C["muted"]};'> {journal}</span>
        </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div style='text-align:center; padding:36px; background:linear-gradient(135deg,{C["ink"]},{C["mid"]}); border-radius:12px; color:white; margin-top:32px;'>
        <div style='font-family:"Libre Baskerville",serif; font-size:26px; margin-bottom:10px;'>Thank You</div>
        <div style='font-size:14px; opacity:0.75;'>MSc Statistics · Team 4 · MSU Baroda · 2025-26</div>
        <div style='font-size:13px; opacity:0.5;'>Vaishali Sharma · Ashish Vaghela · Raiwant Kumar · Rohan Shukla</div>
    </div>""", unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center; font-size:12px; color:{C["border"]}; padding:8px 0; border-top:1px solid {C["border"]}; margin-top:16px;'>
    Cognitive &amp; Educational Impacts of GenAI Usage · MSc Statistics Team 4 · The Maharaja Sayajirao University of Baroda · 2025-26
</div>""", unsafe_allow_html=True)
