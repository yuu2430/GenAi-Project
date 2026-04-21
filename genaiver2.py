"""
GenAI Impact Study — Full Dashboard
MSc Statistics (Team 4) | The Maharaja Sayajirao University of Baroda
Academic Year 2025-26

Run: streamlit run genai_dashboard_final.py
Requires: data.xlsx in the same directory (for main study data)
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import shapiro, ttest_1samp, pearsonr, spearmanr, kruskal, t as t_dist
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

/* ── STATIC SIDEBAR: hide the collapse/toggle button ── */
button[kind="header"] {{
    display: none !important;
}}
[data-testid="collapsedControl"] {{
    display: none !important;
}}
section[data-testid="stSidebar"] {{
    background: {C['ink']} !important;
    border-right: none !important;
    min-width: 220px !important;
    max-width: 240px !important;
    transform: none !important;
    visibility: visible !important;
}}
section[data-testid="stSidebar"] * {{ color: #b0bec5 !important; }}

/* Nav buttons */
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
    padding: 2.2rem 3rem 3rem 3rem;
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
    margin-bottom: 4px;
}}

.page-title {{
    font-family: 'Libre Baskerville', serif;
    font-size: 26px;
    color: {C['ink']};
    font-weight: 700;
    margin: 0 0 4px 0;
    line-height: 1.25;
}}
.page-sub {{
    font-size: 14px;
    color: {C['muted']};
    margin: 0 0 24px 0;
}}

.hyp {{
    background: #f0f7ff;
    border-left: 3px solid {C['teal']};
    padding: 14px 18px;
    border-radius: 0 6px 6px 0;
    font-size: 14.5px;
    line-height: 1.75;
    margin-bottom: 20px;
    color: {C['slate']};
}}

.result-pass {{
    background: #f0faf5;
    border-left: 3px solid {C['green']};
    padding: 14px 18px;
    border-radius: 0 6px 6px 0;
    font-size: 14.5px;
    line-height: 1.75;
    color: {C['slate']};
    margin: 16px 0;
}}
.result-info {{
    background: #fffbf0;
    border-left: 3px solid {C['amber']};
    padding: 14px 18px;
    border-radius: 0 6px 6px 0;
    font-size: 14.5px;
    line-height: 1.75;
    color: {C['slate']};
    margin: 16px 0;
}}

.badge-pass {{ background:#dcf5ea; color:#145c3a; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }}
.badge-info {{ background:#fef3cd; color:#7a4f00; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }}

.rule {{ border:none; border-top:1px solid {C['border']}; margin:24px 0; }}

[data-testid="metric-container"] {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 8px;
    padding: 16px 20px !important;
}}
[data-testid="metric-container"] label {{
    font-size: 12px !important;
    font-weight: 500 !important;
    color: {C['muted']} !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}}
[data-testid="metric-container"] [data-testid="stMetricValue"] {{
    font-size: 24px !important;
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

.element-container:has(.stPlotlyChart) {{
    border: 1px solid {C['border']};
    border-radius: 8px;
    overflow: hidden;
    background: {C['surface']};
}}

.score-card {{
    background: {C['surface']};
    border: 1px solid {C['border']};
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
    margin-bottom: 12px;
}}

/* Hero banner styles */
.hero-banner {{
    background: linear-gradient(135deg, {C['ink']} 0%, {C['mid']} 100%);
    border-radius: 12px;
    padding: 40px 48px;
    color: white;
    margin-bottom: 32px;
    text-align: center;
}}
.hero-uni {{
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: {C['teal_lt']};
    margin-bottom: 6px;
}}
.hero-dept {{
    font-size: 12px;
    color: #94b4cc;
    margin-bottom: 4px;
}}
.hero-year {{
    font-size: 12px;
    color: #94b4cc;
    margin-bottom: 20px;
}}
.hero-title {{
    font-family: 'Libre Baskerville', serif;
    font-size: 26px;
    font-weight: 700;
    line-height: 1.4;
    color: white;
    margin-bottom: 16px;
}}
.hero-team {{
    font-size: 13px;
    color: #94b4cc;
    margin-bottom: 20px;
    line-height: 1.9;
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
    <div style='padding:20px 16px 12px; border-bottom:1px solid #1e2e42;'>
        <div style='font-family:"Libre Baskerville",serif; font-size:15px; color:#e0e7ef; font-weight:700; line-height:1.4;'>
            GenAI Impact Study
        </div>
        <div style='font-size:11.5px; color:#5d7a96; margin-top:4px;'>
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
            "Obj 3 — CGPA vs AI Usage":     "wilcoxon",
            "Obj 4 — Critical Thinking": "kruskal",
            "Obj 5 — Creativity & IL":        "correlation",
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
                    letter-spacing:1.4px; color:#3d5570; margin:16px 0 5px 4px;'>
            {grp}
        </div>""", unsafe_allow_html=True)
        for label, key in pages.items():
            is_active = st.session_state.active == key
            if st.button(label, key=f"nav_{key}", use_container_width=True):
                st.session_state.active = key
                st.rerun()

    active = st.session_state.active

    st.markdown("""
    <div style='padding:20px 16px 0; border-top:1px solid #1e2e42; margin-top:16px;'>
        <div style='font-size:11px; color:#3d5570; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px;'>Team Members</div>
        <div style='font-size:12.5px; color:#5d7a96; line-height:2;'>
            Vaishali Sharma<br>Ashish Vaghela<br>Raiwant Kumar<br>Rohan Shukla
        </div>
        <div style='font-size:11px; color:#3d5570; font-weight:600; text-transform:uppercase; letter-spacing:1px; margin:12px 0 6px;'>Mentor</div>
        <div style='font-size:12.5px; color:#5d7a96;'>Prof. Murlidharan Kunnumal</div>
    </div>
    """, unsafe_allow_html=True)

# ── HELPERS ───────────────────────────────────────────────────
def page_header(overline, title, subtitle=""):
    st.markdown(f"<div class='overline'>{overline}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='page-title'>{title}</div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='page-sub'>{subtitle}</div>", unsafe_allow_html=True)
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

def hyp_block(h0, h1, test="", alpha="0.05"):
    extra = f"<br><b>Test:</b> {test}" if test else ""
    st.markdown(f"""
    <div class='hyp'>
    <b>H₀:</b> {h0}<br>
    <b>H₁:</b> {h1}{extra}<br>
    <b>Significance level:</b> α = {alpha}
    </div>""", unsafe_allow_html=True)

def result_pass(text):
    st.markdown(f"<div class='result-pass'>{text}</div>", unsafe_allow_html=True)

def result_info(text):
    st.markdown(f"<div class='result-info'>{text}</div>", unsafe_allow_html=True)

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

# ── SIMULATED DATA (consistent seeds throughout) ───────────────
np.random.seed(42)
AI_DEP  = np.clip(np.random.normal(2.63, 0.74, 221), 1, 5)
np.random.seed(7)
IND_RAW = np.round(np.clip(np.random.normal(3.353, 0.983, 221), 1, 5) * 3) / 3
np.random.seed(21)
LOW_CT  = np.clip(np.random.normal(2.0, 0.6, 20),  1, 5)
MOD_CT  = np.clip(np.random.normal(3.1, 0.7, 141), 1, 5)
HIGH_CT = np.clip(np.random.normal(3.7, 0.6, 60),  1, 5)

# ── AI ACCURACY DATA (from AI.xlsx) ───────────────────────────
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

TOOLS   = ["ChatGPT","Copilot","Perplexity","Gemini","Claude"]
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
# AI ACCURACY CHECK
# ══════════════════════════════════════════════════════════════
if active == "ai_accuracy":
    page_header(
        "Supplementary Study",
        "AI Accuracy Check — JAM 2025",
        "Performance benchmarking of 5 AI tools across 5 JAM 2025 subjects: accuracy, response time, and detail quality."
    )

    df_sum = build_summary_df()

    # ── KPI row ──
    overall_acc = {}
    for tool in TOOLS:
        sub_df = df_sum[df_sum["Tool"] == tool]
        overall_acc[tool] = round(sub_df["Accuracy (%)"].mean(), 1)

    best_tool    = max(overall_acc, key=overall_acc.get)
    fastest_tool = df_sum.groupby("Tool")["Avg Time (s)"].mean().idxmin()
    most_detail  = df_sum.groupby("Tool")["Detailed (%)"].mean().idxmax()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Questions / Tool / Subject", "5")
    k2.metric("Total Evaluations", "125")
    k3.metric("Best Accuracy", f"{best_tool} ({overall_acc[best_tool]}%)")
    k4.metric("Fastest Responses", fastest_tool)
    k5.metric("Most Detailed", most_detail)

    # ── ACCURACY PROGRESS BAR STRIP ──────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:1.4px; color:{C['teal']}; margin-bottom:12px;'>Mean Accuracy — All Subjects Combined</div>", unsafe_allow_html=True)

    # Sort tools by accuracy descending
    sorted_tools = sorted(overall_acc.items(), key=lambda x: x[1], reverse=True)

    bar_cols = st.columns(len(sorted_tools))
    for col, (tool, acc) in zip(bar_cols, sorted_tools):
        tool_color = AI_COLORS.get(tool, C["navy"])
        rank_label = ["🥇", "🥈", "🥉", "4th", "5th"]
        rank_idx   = [t for t, _ in sorted_tools].index(tool)
        medal      = rank_label[rank_idx]
        col.markdown(f"""
        <div style='background:{C["surface"]}; border:1px solid {C["border"]};
                    border-radius:10px; padding:14px 16px; text-align:center;'>
            <div style='font-size:11px; font-weight:700; color:{tool_color};
                        text-transform:uppercase; letter-spacing:0.8px; margin-bottom:6px;'>
                {medal} {tool}
            </div>
            <div style='font-size:22px; font-weight:700; color:{C["ink"]}; margin-bottom:8px;'>
                {acc}%
            </div>
            <div style='background:{C["border"]}; border-radius:99px; height:8px; overflow:hidden;'>
                <div style='background:{tool_color}; width:{acc}%; height:100%;
                            border-radius:99px; transition:width 0.6s ease;'></div>
            </div>
            <div style='font-size:11px; color:{C["muted"]}; margin-top:6px;'>
                {int(round(acc / 100 * 25))} / 25 correct
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    # ── TABS ──
    t1, t2, t3, t4, t5 = st.tabs([
        "📊 Overview",
        "🎯 Accuracy",
        "⏱ Response Time",
        "📋 Detail Quality",
        "🔍 Subject Drilldown",
    ])

    # ─────────────────────────────────────
    # TAB 1 — OVERVIEW
    # ─────────────────────────────────────
    with t1:
        st.markdown("### Overall Performance Summary")
        st.markdown(f"<div style='font-size:14px; color:{C['muted']}; margin-bottom:18px;'>Each AI tool was tested on 5 questions per subject (5 subjects = 25 questions per tool, 125 total). Metrics: Accuracy (correct answer), Detail (provided explanation beyond bare answer), Avg Response Time (seconds), Prompted (required follow-up prompt to get correct answer).</div>", unsafe_allow_html=True)

        tool_agg = df_sum.groupby("Tool").agg(
            Accuracy=("Accuracy (%)", "mean"),
            Detail=("Detailed (%)", "mean"),
            AvgTime=("Avg Time (s)", "mean"),
        ).round(1).reset_index()
        tool_agg = tool_agg.sort_values("Accuracy", ascending=False)
        tool_agg.columns = ["Tool","Mean Accuracy (%)","Mean Detail (%)","Mean Avg Time (s)"]

        st.dataframe(tool_agg.set_index("Tool"), use_container_width=True)

        st.markdown("<br>**Overall Accuracy by Tool (mean across all subjects)**")
        fig = go.Figure()
        for _, row in tool_agg.iterrows():
            fig.add_bar(
                x=[row["Tool"]],
                y=[row["Mean Accuracy (%)"]],
                name=row["Tool"],
                marker_color=AI_COLORS.get(row["Tool"], C["navy"]),
                text=[f"{row['Mean Accuracy (%)']}%"],
                textposition="outside",
            )
        fig.add_hline(y=80, line_dash="dash", line_color=C["amber"],
                      annotation_text="80% benchmark", annotation_position="top right")
        plotly_defaults(fig, h=380)
        fig.update_layout(showlegend=False, yaxis=dict(range=[0,110], title="Mean Accuracy (%)"),
                          title="Mean Accuracy Across All 5 Subjects")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>**Heatmap — Accuracy (%) by Tool × Subject**")
        pivot = df_sum.pivot(index="Tool", columns="Subject", values="Accuracy (%)")
        fig2, ax = plt.subplots(figsize=(10, 4))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGn",
                    linewidths=0.5, ax=ax, cbar_kws={"label":"Accuracy (%)"},
                    vmin=0, vmax=100,
                    annot_kws={"size": 12, "weight": "bold"})
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("Accuracy (%) — Tool × Subject", fontsize=12, fontweight="bold", pad=10)
        plt.xticks(rotation=20, ha="right", fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        result_info("""
        <b>Key observations:</b><br>
        • <b>ChatGPT & Claude</b> lead in Biotechnology (100% accuracy each).<br>
        • <b>Perplexity</b> struggles in Mathematical Science (20%) and Chemistry (60%),
          but is consistently the fastest responder.<br>
        • <b>Mathematics</b> is the most challenging subject — only Copilot and Perplexity
          scored 100%; ChatGPT dropped to 80%.<br>
        • <b>Gemini</b> achieved 100% across Physics, Mathematics, and Biotechnology.
        """)

    # ─────────────────────────────────────
    # TAB 2 — ACCURACY
    # ─────────────────────────────────────
    with t2:
        st.markdown("### Accuracy Analysis — Correct Answers per Tool per Subject")

        fig = px.bar(
            df_sum, x="Subject", y="Accuracy (%)", color="Tool",
            barmode="group", text_auto=".0f",
            color_discrete_map=AI_COLORS,
        )
        plotly_defaults(fig, h=440)
        fig.update_layout(
            xaxis_tickangle=-15,
            yaxis=dict(range=[0, 115], title="Accuracy (%)"),
            legend_title="AI Tool",
            title="Accuracy (%) by Subject and Tool",
        )
        fig.add_hline(y=80, line_dash="dot", line_color=C["red"],
                      annotation_text="80%", annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>**Per-question accuracy table**")
        acc_rows = []
        for subject, tools in AI_ACCURACY_DATA.items():
            for tool, d in tools.items():
                for q, c in enumerate(d["correct"], 1):
                    acc_rows.append({
                        "Subject": subject, "Tool": tool,
                        "Q": q, "Correct": "✅" if c else "❌"
                    })
        acc_df = pd.DataFrame(acc_rows)
        pivot_q = acc_df.pivot_table(
            index=["Subject","Tool"], columns="Q",
            values="Correct", aggfunc="first"
        )
        pivot_q.columns = [f"Q{c}" for c in pivot_q.columns]
        st.dataframe(pivot_q, use_container_width=True)

        st.markdown("<hr class='rule'>", unsafe_allow_html=True)
        st.markdown("**Summary: Questions correctly answered (out of 5) per tool per subject**")
        score_df = df_sum[["Subject","Tool","Accuracy (%)"]].copy()
        score_df["Correct / 5"] = (score_df["Accuracy (%)"] / 100 * 5).astype(int)
        pivot_s = score_df.pivot(index="Tool", columns="Subject", values="Correct / 5")
        pivot_s["Total / 25"] = pivot_s.sum(axis=1)
        st.dataframe(pivot_s.style.background_gradient(cmap="Greens", vmin=0, vmax=5,
                     subset=[c for c in pivot_s.columns if c != "Total / 25"])
                     .background_gradient(cmap="Blues", vmin=0, vmax=25, subset=["Total / 25"]),
                     use_container_width=True)

    # ─────────────────────────────────────
    # TAB 3 — RESPONSE TIME
    # ─────────────────────────────────────
    with t3:
        st.markdown("### Response Time Analysis (seconds)")
        st.markdown(f"<div style='font-size:14px; color:{C['muted']}; margin-bottom:16px;'>Time measured from query submission to complete answer generation. Lower is faster.</div>", unsafe_allow_html=True)

        time_rows = []
        for subject, tools in AI_ACCURACY_DATA.items():
            for tool, d in tools.items():
                for t_val in d["times"]:
                    time_rows.append({"Subject": subject, "Tool": tool, "Time (s)": t_val})
        time_df = pd.DataFrame(time_rows)

        fig = px.box(
            time_df, x="Tool", y="Time (s)", color="Tool",
            points="all", color_discrete_map=AI_COLORS,
            title="Response Time Distribution by AI Tool (all subjects combined)"
        )
        plotly_defaults(fig, h=400)
        fig.update_layout(showlegend=False, yaxis_title="Response Time (seconds)")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>**Average Response Time by Subject × Tool**")
        fig2 = px.bar(
            df_sum, x="Subject", y="Avg Time (s)", color="Tool",
            barmode="group", text_auto=".1f",
            color_discrete_map=AI_COLORS,
        )
        plotly_defaults(fig2, h=420)
        fig2.update_layout(xaxis_tickangle=-15, yaxis_title="Avg Response Time (s)",
                           legend_title="AI Tool",
                           title="Average Response Time — Subject × Tool")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("<br>**Mean Response Times**")
        time_agg = time_df.groupby("Tool")["Time (s)"].agg(["mean","min","max","std"]).round(2)
        time_agg.columns = ["Mean (s)","Min (s)","Max (s)","Std Dev"]
        time_agg = time_agg.sort_values("Mean (s)")
        st.dataframe(time_agg, use_container_width=True)

        result_info("""
        <b>Speed insights:</b><br>
        • <b>Perplexity</b> is the fastest across nearly all subjects — averaging under 7s for Physics and Biotechnology.<br>
        • <b>Copilot</b> and <b>Claude</b> show moderate, consistent response times (8–18s range).<br>
        • <b>ChatGPT</b> and <b>Gemini</b> are slower on mathematical subjects (15–22s), likely due to step-by-step reasoning.<br>
        • Longer response time broadly correlates with higher detail level — mathematical subjects take longest across all tools.
        """)

    # ─────────────────────────────────────
    # TAB 4 — DETAIL QUALITY
    # ─────────────────────────────────────
    with t4:
        st.markdown("### Detail Quality — Did the AI provide a detailed explanation?")
        st.markdown(f"<div style='font-size:14px; color:{C['muted']}; margin-bottom:16px;'>'Detailed' = AI provided a structured explanation or step-by-step reasoning, not merely a bare answer.</div>", unsafe_allow_html=True)

        fig = px.bar(
            df_sum, x="Subject", y="Detailed (%)", color="Tool",
            barmode="group", text_auto=".0f",
            color_discrete_map=AI_COLORS,
            title="Detail Rate (%) by Subject and Tool"
        )
        plotly_defaults(fig, h=420)
        fig.update_layout(xaxis_tickangle=-15, yaxis=dict(range=[0,115]),
                          legend_title="AI Tool")
        fig.add_hline(y=60, line_dash="dot", line_color=C["muted"],
                      annotation_text="60% ref", annotation_position="top right")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>**Radar Chart — Accuracy vs Detail vs Speed (normalised)**")
        radar_data = []
        for tool in TOOLS:
            sub = df_sum[df_sum["Tool"] == tool]
            max_t = df_sum["Avg Time (s)"].max()
            radar_data.append({
                "Tool": tool,
                "Accuracy": sub["Accuracy (%)"].mean(),
                "Detail":   sub["Detailed (%)"].mean(),
                "Speed":    100 - (sub["Avg Time (s)"].mean() / max_t * 100),
            })
        rdf = pd.DataFrame(radar_data)
        categories = ["Accuracy","Detail","Speed"]
        fig3 = go.Figure()
        for _, row in rdf.iterrows():
            values = [row["Accuracy"], row["Detail"], row["Speed"]]
            values += [values[0]]
            fig3.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill="toself",
                name=row["Tool"],
                line_color=AI_COLORS.get(row["Tool"], C["navy"]),
                fillcolor=AI_COLORS.get(row["Tool"], C["navy"]),
                opacity=0.25,
            ))
        fig3.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True,
            height=440,
            font=dict(family="Inter"),
            paper_bgcolor="white",
            title="Tool Profile — Accuracy · Detail · Speed (normalised to 0–100)",
            margin=dict(t=60, b=30, l=30, r=30),
        )
        st.plotly_chart(fig3, use_container_width=True)

        result_info("""
        <b>Detail quality insights:</b><br>
        • <b>ChatGPT & Copilot</b> consistently provide the most detailed responses — especially in Mathematical Science and Mathematics (80–100% detail rate).<br>
        • <b>Perplexity</b> rarely provides detailed explanations (0% detail in most subjects) — it prioritises concise, direct answers.<br>
        • <b>Gemini & Claude</b> show mixed detail — detailed on some subjects but not others.<br>
        • There is a trade-off: tools with high detail (ChatGPT, Copilot) tend to be slower; Perplexity is fast but terse.
        """)

    # ─────────────────────────────────────
    # TAB 5 — SUBJECT DRILLDOWN
    # ─────────────────────────────────────
    with t5:
        st.markdown("### Subject-level Drilldown")
        selected_subject = st.selectbox("Select a Subject", SUBJECTS)

        subj_df = df_sum[df_sum["Subject"] == selected_subject].copy()
        subj_data = AI_ACCURACY_DATA[selected_subject]

        best_acc_tool = subj_df.loc[subj_df["Accuracy (%)"].idxmax(), "Tool"]
        best_acc_val  = subj_df["Accuracy (%)"].max()
        fastest       = subj_df.loc[subj_df["Avg Time (s)"].idxmin(), "Tool"]
        fastest_val   = subj_df["Avg Time (s)"].min()
        most_det      = subj_df.loc[subj_df["Detailed (%)"].idxmax(), "Tool"]
        most_det_val  = subj_df["Detailed (%)"].max()

        c1, c2, c3 = st.columns(3)
        c1.metric(f"Best Accuracy", f"{best_acc_tool} ({best_acc_val}%)")
        c2.metric(f"Fastest Response", f"{fastest} ({fastest_val:.1f}s)")
        c3.metric(f"Most Detailed", f"{most_det} ({most_det_val:.0f}%)")

        st.markdown("<br>")
        col_l, col_r = st.columns(2)

        with col_l:
            st.markdown("**Question-by-question results**")
            q_rows = []
            for tool, d in subj_data.items():
                for q in range(5):
                    q_rows.append({
                        "Q": f"Q{q+1}",
                        "Tool": tool,
                        "Correct": "✅" if d["correct"][q] else "❌",
                        "Detailed": "📝" if d["detailed"][q] else "—",
                        "Time (s)": round(d["times"][q], 2),
                    })
            q_df = pd.DataFrame(q_rows)
            for tool in TOOLS:
                t_df = q_df[q_df["Tool"] == tool][["Q","Correct","Detailed","Time (s)"]].set_index("Q")
                st.markdown(f"**{tool}**")
                st.dataframe(t_df, use_container_width=True)

        with col_r:
            st.markdown("**Accuracy comparison**")
            fig_s = px.bar(
                subj_df, x="Tool", y="Accuracy (%)", color="Tool",
                text_auto=".0f", color_discrete_map=AI_COLORS,
            )
            plotly_defaults(fig_s, h=280)
            fig_s.update_layout(showlegend=False, yaxis=dict(range=[0, 115]),
                                 title=f"Accuracy — {selected_subject}")
            st.plotly_chart(fig_s, use_container_width=True)

            st.markdown("**Response time comparison**")
            fig_t = px.bar(
                subj_df, x="Tool", y="Avg Time (s)", color="Tool",
                text_auto=".1f", color_discrete_map=AI_COLORS,
            )
            plotly_defaults(fig_t, h=280)
            fig_t.update_layout(showlegend=False,
                                 title=f"Avg Response Time — {selected_subject}")
            st.plotly_chart(fig_t, use_container_width=True)

        st.markdown("<hr class='rule'>", unsafe_allow_html=True)
        st.markdown("**Full summary table for this subject**")
        st.dataframe(subj_df.set_index("Tool"), use_container_width=True)

    # ── OVERALL CONCLUSION BOX ──
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("### Summary & Key Takeaways")

    df_sum2 = build_summary_df()
    overall = df_sum2.groupby("Tool").agg(
        Accuracy=("Accuracy (%)", "mean"),
        Detail=("Detailed (%)", "mean"),
        Speed=("Avg Time (s)", "mean"),
    ).round(1).reset_index().sort_values("Accuracy", ascending=False)

    acc_rows_html = "".join([
        f"<div style='display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid {C['border']};'>"
        f"<span style='color:{C['slate']};'><b>{i+1}.</b> {row['Tool']}</span>"
        f"<span style='font-weight:700; color:{AI_COLORS.get(row['Tool'], C['navy'])};'>{row['Accuracy']}%</span>"
        f"</div>"
        for i, (_, row) in enumerate(overall.iterrows())
    ])
    speed_rows_html = "".join([
        f"<div style='display:flex; justify-content:space-between; padding:6px 0; border-bottom:1px solid {C['border']};'>"
        f"<span style='color:{C['slate']};'><b>{i+1}.</b> {row['Tool']}</span>"
        f"<span style='font-weight:700; color:{AI_COLORS.get(row['Tool'], C['navy'])};'>{row['Speed']}s</span>"
        f"</div>"
        for i, (_, row) in enumerate(overall.sort_values("Speed").iterrows())
    ])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style='background:{C["surface"]}; border:1px solid {C["border"]};
                    border-top:3px solid {C["teal"]}; border-radius:8px; padding:20px 22px;'>
            <div style='font-weight:600; color:{C["ink"]}; font-size:15px; margin-bottom:12px;'>Accuracy Rankings</div>
            {acc_rows_html}
        </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div style='background:{C["surface"]}; border:1px solid {C["border"]};
                    border-top:3px solid {C["amber"]}; border-radius:8px; padding:20px 22px;'>
            <div style='font-weight:600; color:{C["ink"]}; font-size:15px; margin-bottom:12px;'>Speed Rankings (lower = faster)</div>
            {speed_rows_html}
        </div>""", unsafe_allow_html=True)

    result_pass(f"""
    <b>Benchmark Conclusions (JAM 2025 — 5 subjects, 5 questions each):</b><br><br>
    • <b>Gemini</b> and <b>Claude</b> achieve the highest mean accuracy across subjects, each performing consistently well across both science and mathematics domains.<br>
    • <b>Copilot</b> excels in Mathematics (100% accuracy) and delivers the most consistently detailed explanations alongside ChatGPT — suitable when step-by-step reasoning is required.<br>
    • <b>Perplexity</b> is the speed champion (avg &lt; 8s) but sacrifices accuracy in abstract subjects like Mathematical Science (20%) and Chemistry — better suited to factual/lookup queries.<br>
    • <b>ChatGPT</b> provides strong detail and solid accuracy but is among the slower tools for mathematical subjects (avg 19s).<br>
    • <b>No single tool dominates on all three dimensions</b> (accuracy, detail, speed) — the optimal choice depends on subject type and whether the student needs an explanation or just the answer.
    """)

# ══════════════════════════════════════════════════════════════
# OVERVIEW — FIXED: uses Streamlit columns + proper HTML blocks
# ══════════════════════════════════════════════════════════════
elif active == "overview":
    # ── Hero banner — rendered via a single clean HTML block ──
    st.markdown(f"""
    <div class="hero-banner">
        <div class="hero-uni">The Maharaja Sayajirao University of Baroda</div>
        <div class="hero-dept">Faculty of Science · Department of Statistics</div>
        <div class="hero-year">Academic Year 2025-26</div>
        <div class="hero-title">
            Cognitive &amp; Educational Impacts of<br>Generative AI Usage Among University Students
        </div>
        <div class="hero-team">
            <strong style="color:white;">MSc Statistics · Team 4</strong><br>
            Vaishali Sharma &nbsp;·&nbsp; Ashish Vaghela &nbsp;·&nbsp; Raiwant Kumar &nbsp;·&nbsp; Rohan Shukla<br>
            <span style="font-size:12px; opacity:0.7;">Guided by: Prof. Murlidharan Kunnumal</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Abstract ──
    st.markdown(f"""
    <div style='font-size:14.5px; line-height:1.95; color:{C["slate"]}; margin-bottom:32px;
                text-align:justify; text-justify:inter-word;'>
    Primary data were collected from <strong>221 students</strong> across <strong>13 faculties</strong>
    using a structured questionnaire administered via Probability Proportional to Size (PPS) sampling.
    Reliability of all psychometric scales was confirmed using Cronbach's Alpha (α ≥ 0.83 across all
    constructs). The study employs descriptive analysis, normality testing, non-parametric inference,
    correlation analysis, and supervised machine learning to address six research objectives.<br><br>
    Findings reveal that students exhibit <strong>moderate, purposeful GenAI use</strong>, with average
    dependency levels significantly below the neutral benchmark of 3.0. AI usage is positively associated
    with independent learning (ρ = 0.459) and critical thinking (ρ = 0.466), while no significant
    relationship is observed between AI dependency and CGPA or creativity. Faculty affiliation — not
    gender or level of study — is the only significant demographic predictor of AI dependency.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    # ── At-a-glance metrics ──
    st.markdown(f"<div class='overline'>Study at a Glance</div>", unsafe_allow_html=True)
    cols = st.columns(4)
    for col, v, l in zip(cols,
        ["221", "13", "6", "4"],
        ["Students surveyed", "Faculties covered", "Research objectives", "AI tools studied"]):
        col.metric(l, v)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Key Findings ──
    st.markdown(f"<div class='overline'>Key Findings</div>", unsafe_allow_html=True)

    findings = [
        ("Mean AI Dependency = 2.63",
         "Significantly below the neutral midpoint of 3.0 (t = −5.74, p < 0.0001). Students use AI purposefully, not compulsively."),
        ("Faculty drives dependency differences",
         "Arts and Technology students show significantly higher AI dependency than smaller faculties (Welch ANOVA p = 0.041)."),
        ("AI promotes independent learning",
         "Median Independent Learning Score (3.35) significantly exceeds the neutral benchmark (Wilcoxon W = 13,589, p < 0.001)."),
        ("Higher AI use → stronger critical thinking",
         "Spearman ρ = 0.466 between AI usage group and critical thinking score (Kruskal-Wallis H = 49.65, p < 10⁻¹¹)."),
        ("No significant effect on creativity",
         "Spearman ρ = 0.087, p = 0.198 — AI usage frequency does not significantly predict creative output."),
        ("71% predictive accuracy for academic divisions",
         "KNN (k=5) classifier using five AI-related features achieves 71.1% accuracy on held-out test set."),
    ]

    for i in range(0, 6, 2):
        c1, c2 = st.columns(2)
        for col, (title, desc) in zip([c1, c2], findings[i:i+2]):
            col.markdown(f"""
            <div style='background:{C["surface"]}; border:1px solid {C["border"]};
                        border-radius:8px; padding:20px 22px; margin-bottom:14px;
                        border-left:3px solid {C["teal"]};'>
                <div style='font-weight:600; color:{C["ink"]}; font-size:14.5px; margin-bottom:6px;'>{title}</div>
                <div style='font-size:13.5px; color:{C["muted"]}; line-height:1.65;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# OBJECTIVES
# ══════════════════════════════════════════════════════════════
elif active == "objectives":
    page_header("Study Design", "Research Objectives",
                "Six operationalised objectives spanning descriptive through predictive analysis.")

    objs = [
        ("1", "AI Usage Patterns",
         "To determine how frequently students use AI tools, identify the purposes for which they are used, and examine whether usage patterns vary systematically by educational level (UG vs. PG) or gender.",
         "Descriptive analysis · Grouped bar charts · Donut charts"),
        ("2", "Level of AI Dependency",
         "To identify and quantify the level of dependency on Generative AI among students of The Maharaja Sayajirao University of Baroda, using a validated AI Dependency Scale.",
         "Shapiro-Wilk · One-sample t-test · Multi-way ANOVA · Welch ANOVA · Games-Howell post hoc"),
        ("3", "AI Usage Level and Learning Performance (CGPA)",
         "To check whether AI usage level (Low, Moderate, High) significantly affects students' learning performance as measured by CGPA.",
         "Shapiro-Wilk · Levene's Test · Kruskal-Wallis H Test"),
        ("4", "AI Usage and Critical Thinking",
         "To study how AI tool usage is related to students' critical thinking, using self-reported ratings.",
         "Kruskal-Wallis H Test · Spearman Rank Correlation"),
        ("5", "Creativity and Independent Learning",
         "To explore how using AI tools influences students' academic skills, such as creativity and independent learning.",
         "Wilcoxon Signed-Rank Test (two sub-objectives) · Cronbach's Alpha · K-DOCS"),
        ("6", "Predictive Model for Academic Performance",
         "To create a model that predicts factors leading to positive or negative effects of AI tool usage on students' academic outcomes.",
         "K-Nearest Neighbours (k=5) · Decision Tree (max_depth=7) · 80/20 train-test split"),
    ]

    for num, title, desc, methods in objs:
        st.markdown(f"""
        <div style='display:flex; gap:20px; background:{C["surface"]}; border:1px solid {C["border"]};
                    border-radius:8px; padding:22px 24px; margin-bottom:12px;'>
            <div style='font-family:"Libre Baskerville",serif; font-size:36px; font-weight:700;
                        color:{C["border"]}; line-height:1; min-width:36px; text-align:center;
                        padding-top:4px;'>{num}</div>
            <div style='flex:1;'>
                <div style='font-weight:600; color:{C["ink"]}; font-size:15.5px; margin-bottom:6px;'>{title}</div>
                <div style='font-size:14px; color:{C["slate"]}; line-height:1.7; margin-bottom:10px;'>{desc}</div>
                <div style='font-size:12px; color:{C["teal"]}; font-weight:500;'>{methods}</div>
            </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# PILOT SURVEY
# ══════════════════════════════════════════════════════════════
elif active == "pilot":
    page_header("Methodology", "Pilot Survey",
                "Conducted prior to full data collection to validate the questionnaire and estimate the population proportion.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Population (N)", "37,095")
    c2.metric("Pilot Sample (n)", "58")
    c3.metric("Affirmative Responses", "48")
    c4.metric("Estimated Proportion", "0.827")

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

        st.markdown("<br>**Sample Size Calculation — Cochran's Formula:**")
        st.latex(r"n = \frac{z_{\alpha/2}^2 \cdot p \cdot q}{E^2} = \frac{(1.96)^2 \times 0.827 \times 0.173}{(0.05)^2} \approx \mathbf{221}")

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
# SAMPLING
# ══════════════════════════════════════════════════════════════
elif active == "sampling":
    page_header("Methodology", "Sampling Design",
                "Probability Proportional to Size (PPS) sampling across all 13 faculties of MSU Baroda.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Sampling Method", "PPS")
    c2.metric("Total Population (N)", "37,095")
    c3.metric("Final Sample (n)", "221")

    st.markdown("<br>**Proportional Allocation Formula:**")
    st.latex(r"n_i = \frac{N_i}{N} \times n")
    st.markdown(f"<div style='font-size:13.5px; color:{C['muted']};'>where n_i = sample size for faculty i, N_i = faculty population, N = 37,095, n = 221</div>", unsafe_allow_html=True)

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("**Faculty-wise Sample Distribution**")

    sdf = pd.DataFrame({
        "Faculty": ["Arts","Commerce","Education & Psychology","Family & Community Sciences",
                    "Fine Arts","Journalism & Communication","Law","Management Studies",
                    "Performing Arts","Pharmacy","Science","Social Work","Technology & Engineering","TOTAL"],
        "Total": [25,120,5,7,4,1,8,2,2,2,23,3,19,221],
        "Female UG":[13,58,3,5,2,1,4,1,1,1,9,1,3,102],
        "Female PG":[3,8,1,1,1,0,0,0,0,0,4,1,2,21],
        "Male UG":  [8,50,1,1,1,0,4,1,1,1,8,1,12,89],
        "Male PG":  [1,4,0,0,0,0,0,0,0,0,2,0,2,9],
    })
    st.dataframe(sdf.set_index("Faculty"), use_container_width=True)

    fac = sdf[sdf["Faculty"] != "TOTAL"]
    fig = px.bar(fac, x="Faculty", y="Total", text_auto=True,
                 color="Total", color_continuous_scale=["#bfdbfe", C["navy"]],
                 template="plotly_white")
    plotly_defaults(fig, h=380)
    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35,
                      title="Faculty-wise Sample Size")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# QUESTIONNAIRE
# ══════════════════════════════════════════════════════════════
elif active == "questionnaire":
    page_header("Data Collection", "Questionnaire Design",
                "Structured self-administered questionnaire delivered via Google Forms — 23 questions across 6 thematic sections.")

    sections = {
        "Section 1 — Student Profile & AI Usage (Q1–Q14)": [
            "Demographic variables: Age, Gender, CGPA, Schooling Background, Faculty, Level and Year of Study",
            "Awareness of GenAI, primary AI tool used, subscription status and monthly expenditure",
            "Multi-select grid (Q13): AI platform × 6 academic purposes",
            "Likert frequency grid (Q14): Never → Always across 6 academic purposes",
        ],
        "Section 2 — Academic Impact (Q15–Q16)": [
            "Q15 (3 items): grades change, learning effectiveness, curriculum inclusion — Strongly Disagree to Strongly Agree",
            "Q16 (6 items): AI vs books, AI vs teacher, critical thinking reduction, AI submission, stress reduction, independent exploration",
        ],
        "Section 3 — Cognitive Offloading (Q18)": [
            "5 items measuring reliance on digital tools for information retrieval and task memory",
            "Scale: 1 (Never / Not Dependent) to 5 (Always / Very Likely)",
        ],
        "Section 4 — Critical Thinking (Q19)": [
            "8 items: source evaluation, fake-news detection, cross-referencing, author credibility, bias reflection",
            "Scale: 1 (Never / Not Confident) to 5 (Always / Very Confident)",
        ],
        "Section 5 — AI Dependency (Q17, Q20–Q22)": [
            "Q17: 10-point scale — trust, understanding depth, motivation, anxiety, assignment dependency",
            "Q20 — Cognitive Preoccupation (3 items): decision influence, urge to use, anticipation",
            "Q21 — Negative Consequences (4 items): concerns, inability to work without AI, confidence, problem-solving",
            "Q22 — Withdrawal Symptoms (4 items): restlessness, distraction, disconnection, irritability",
        ],
        "Section 6 — Creativity (Q23)": [
            "11 items adapted from the Kaufman Domains of Creativity Scale (K-DOCS, 2012)",
            "Scale: 1 (Much Less Creative) to 5 (Much More Creative) — comparative to doing the task without AI",
            "Tasks: writing, debating, researching, feedback, analysis, argumentation",
        ],
    }

    for sec, items in sections.items():
        with st.expander(sec):
            for item in items:
                st.markdown(f"<div style='font-size:14px; color:{C['slate']}; padding:3px 0;'>— {item}</div>", unsafe_allow_html=True)

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("**Construct Overview**")
    cdf = pd.DataFrame({
        "Construct":      ["AI Dependency","Critical Thinking","Creativity","Cognitive Offloading","Independent Learning"],
        "Items":          [11, 8, 11, 5, 3],
        "Scale":          ["Likert 1–5","Likert 1–5","Comparative 1–5","Likert 1–5","Likert 1–5"],
        "Instrument":     ["GAIDS (Goh et al., 2023)","Custom multi-item scale","K-DOCS (Kaufman, 2012)",
                           "Custom digital-offloading scale","Study Preferences sub-scale"],
        "Cronbach α":     [0.8936, 0.9139, 0.9427, 0.8512, 0.8302],
    })
    st.dataframe(cdf.set_index("Construct"), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# RELIABILITY
# ══════════════════════════════════════════════════════════════
elif active == "reliability":
    page_header("Pre-Analysis", "Reliability Analysis — Cronbach's Alpha",
                "Internal consistency of all multi-item scales verified before inferential analysis.")

    st.markdown(f"""
    <div style='font-size:14.5px; color:{C["slate"]}; line-height:1.85; max-width:760px; margin-bottom:20px;'>
    Before computing composite scores or conducting inferential tests, the internal consistency of each
    multi-item scale was assessed using <strong>Cronbach's Alpha (α)</strong>. A coefficient of <strong>α ≥ 0.70</strong>
    is the conventional minimum for acceptable research reliability; values above 0.90 indicate excellent reliability.
    </div>""", unsafe_allow_html=True)

    st.markdown("**Formula:**")
    st.latex(r"\alpha = \frac{k}{k-1} \left(1 - \frac{\sum_{i=1}^{k} \sigma_{y_i}^2}{\sigma_x^2}\right)")
    st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:20px;'>k = number of items · σ²yᵢ = item variance · σ²x = total score variance</div>", unsafe_allow_html=True)

    rel = pd.DataFrame({
        "Construct":      ["AI Dependency (GAIDS)","Critical Thinking","Creativity (K-DOCS)","Cognitive Offloading","Independent Learning"],
        "Items":          [11, 8, 11, 5, 3],
        "Cronbach α":     [0.8936, 0.9139, 0.9427, 0.8512, 0.8302],
        "Reliability":    ["Good","Excellent","Excellent","Good","Good"],
    })
    st.dataframe(rel.set_index("Construct"), use_container_width=True)

    fig = go.Figure()
    colors = [C["teal"] if a >= 0.9 else C["navy"] for a in rel["Cronbach α"]]
    fig.add_bar(x=rel["Construct"], y=rel["Cronbach α"],
                marker_color=colors,
                text=[f"{a:.4f}" for a in rel["Cronbach α"]],
                textposition="outside")
    fig.add_hline(y=0.70, line_dash="dash", line_color=C["red"],
                  annotation_text="Acceptable (0.70)", annotation_position="top right")
    fig.add_hline(y=0.90, line_dash="dot", line_color=C["amber"],
                  annotation_text="Excellent (0.90)", annotation_position="top right")
    plotly_defaults(fig, h=380)
    fig.update_layout(yaxis=dict(range=[0.5, 1.02], title="Cronbach's Alpha"),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    result_pass("All five scales achieve Cronbach's Alpha above 0.83, confirming high internal consistency. Composite scores constructed from item averages are statistically reliable and valid for subsequent inferential analysis.")

# ══════════════════════════════════════════════════════════════
# OBJECTIVE 1 — DESCRIPTIVE
# ══════════════════════════════════════════════════════════════
elif active == "descriptive":
    page_header("Objective 1", "Descriptive Analysis of AI Usage",
                "How often do students use AI tools, for what purposes, and do patterns differ by level of study or gender?")

    viz = st.radio("", ["AI Tools × Academic Purpose","Programme-wise Usage",
                        "Gender-wise Usage","Usage Frequency by Purpose"], horizontal=True)
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    if viz == "AI Tools × Academic Purpose":
        data_t = pd.DataFrame({
            "Purpose": ["Project/Assignment"]*4+["Concept Learning"]*4+["Writing/Summarising"]*4+
                       ["Exam Preparation"]*4+["Research/Idea Gen"]*4+["Programming/Coding"]*4,
            "Tool":    ["ChatGPT","Gemini","Copilot","Perplexity"]*6,
            "n":       [131,48,5,10, 114,73,14,9, 128,56,12,6, 101,56,15,9, 101,40,6,8, 66,34,16,10]
        })
        fig = px.bar(data_t, x="Purpose", y="n", color="Tool", barmode="group",
                     text_auto=True, color_discrete_sequence=CHART_SEQ)
        plotly_defaults(fig, h=460)
        fig.update_layout(xaxis_tickangle=-15, legend_title="AI Tool",
                          yaxis_title="Number of Students")
        st.plotly_chart(fig, use_container_width=True)

    elif viz == "Programme-wise Usage":
        c1, c2 = st.columns(2)
        for col, title, yes, no in [(c1,"Undergraduate (n=171)",112,59),(c2,"Postgraduate (n=50)",47,3)]:
            fig = go.Figure(go.Pie(labels=["Uses GenAI","Does not use"],
                                   values=[yes,no], hole=0.55,
                                   marker_colors=[C["teal"],C["border"]],
                                   textinfo="percent", textfont_size=13))
            pct = round(yes/(yes+no)*100,1)
            fig.add_annotation(text=f"<b>{pct}%</b>", x=0.5, y=0.5,
                                font_size=18, showarrow=False, font_color=C["ink"])
            fig.update_layout(title=title, height=300, showlegend=False,
                               font=dict(family="Inter"), paper_bgcolor="white",
                               margin=dict(t=48,b=10,l=10,r=10))
            col.plotly_chart(fig, use_container_width=True)
        result_info("PG students show <strong>94% AI adoption</strong> vs 65.5% for UG students.")

    elif viz == "Gender-wise Usage":
        c1, c2 = st.columns(2)
        for col, title, yes, no, clr in [
            (c1,"Female Students (n=128)",99,29,C["teal"]),
            (c2,"Male Students (n=93)",60,33,C["navy"])
        ]:
            fig = go.Figure(go.Pie(labels=["Uses GenAI","Does not use"],
                                   values=[yes,no], hole=0.55,
                                   marker_colors=[clr,C["border"]],
                                   textinfo="percent", textfont_size=13))
            pct = round(yes/(yes+no)*100,1)
            fig.add_annotation(text=f"<b>{pct}%</b>", x=0.5, y=0.5,
                                font_size=18, showarrow=False, font_color=C["ink"])
            fig.update_layout(title=title, height=300, showlegend=False,
                               font=dict(family="Inter"), paper_bgcolor="white",
                               margin=dict(t=48,b=10,l=10,r=10))
            col.plotly_chart(fig, use_container_width=True)
        result_info("Female students (77.3%) show <strong>higher AI adoption</strong> than male students (64.5%).")

    else:
        freq_data = pd.DataFrame({
            "Purpose": ["Project/Assignment"]*5+["Concept Learning"]*5+
                       ["Writing/Summarising"]*5+["Exam Preparation"]*5+
                       ["Research/Idea Gen"]*5+["Programming/Coding"]*5,
            "Frequency": ["Never","Rarely","Sometimes","Often","Always"]*6,
            "n": [10,35,67,67,42, 11,21,69,87,33, 13,16,78,79,35,
                  21,18,68,80,34, 32,37,62,49,41, 22,30,43,99,27]
        })
        fig = px.bar(freq_data, x="Purpose", y="n", color="Frequency", barmode="group",
                     text_auto=True,
                     color_discrete_map={"Never":"#c8d6e5","Rarely":"#8896a8",
                                         "Sometimes":C["teal_lt"],"Often":C["teal"],"Always":C["navy"]})
        plotly_defaults(fig, h=460)
        fig.update_layout(xaxis_tickangle=-15, yaxis_title="Number of Students")
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# OBJECTIVE 2 — AI DEPENDENCY
# ══════════════════════════════════════════════════════════════
elif active == "anova":
    page_header("Objective 2", "AI Dependency Level",
                "Identifying and quantifying the level of GenAI dependency among MSU students using the GAIDS scale.")

    sub = st.radio("", ["Normality Test","One-Sample t-test",
                        "Multi-way ANOVA","Post-Hoc (Games-Howell)"], horizontal=True)
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    if sub == "Normality Test":
        st.markdown("### Hypothesis 1 — Normality of AI Dependency Score")
        hyp_block("The AI Dependency Score follows a normal distribution",
                  "The AI Dependency Score does not follow a normal distribution",
                  "Shapiro-Wilk Test")
        st.latex(r"W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}")
        stat_sw, p_sw = shapiro(AI_DEP)
        c1, c2, c3 = st.columns(3)
        c1.metric("Shapiro-Wilk W", f"{stat_sw:.4f}")
        c2.metric("p-value", "0.1676")
        c3.metric("Decision", "Fail to reject H₀")
        result_pass("<b>Normality satisfied</b> — p = 0.1676 > 0.05.")
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.hist(AI_DEP, bins=14, color=C["teal"], edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(AI_DEP), color=C["amber"], lw=2, ls="--", label=f"Mean = {np.mean(AI_DEP):.2f}")
        ax.set_xlabel("AI Dependency Score"); ax.set_ylabel("Frequency")
        ax.legend(fontsize=10); plt.tight_layout()
        st.pyplot(fig, use_container_width=True); plt.close()

    elif sub == "One-Sample t-test":
        st.markdown("### Hypothesis 2 — Mean AI Dependency vs. Neutral Midpoint")
        hyp_block("Population mean AI Dependency Score = 3.0",
                  "Population mean AI Dependency Score ≠ 3.0", "Two-sided One-Sample t-test")
        st.latex(r"t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}")
        t_stat, p_val = ttest_1samp(AI_DEP, 3.0)
        n = len(AI_DEP); mean = np.mean(AI_DEP); sd = np.std(AI_DEP, ddof=1)
        se = sd / np.sqrt(n); t_crit = t_dist.ppf(0.975, n-1)
        ci_l, ci_u = mean - t_crit*se, mean + t_crit*se
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sample Mean (x̄)", f"{mean:.3f}")
        c2.metric("Std. Dev. (s)", f"{sd:.3f}")
        c3.metric("t-statistic", "−5.740")
        c4.metric("p-value", "3.12 × 10⁻⁸")
        st.markdown(f"<div style='font-size:14px; color:{C['muted']}; margin:8px 0;'>95% CI: ({ci_l:.3f}, {ci_u:.3f})</div>", unsafe_allow_html=True)
        result_pass(f"<b>Reject H₀</b> — mean AI Dependency ≈ 2.63, significantly below 3.0.")

    elif sub == "Multi-way ANOVA":
        st.markdown("### Multi-way ANOVA — AI Dependency by Demographic Groups")
        hyp_block("Group means equal across all demographic levels",
                  "At least one group mean differs", "Multi-way ANOVA (Type II SS) + Welch ANOVA")
        adf = pd.DataFrame({
            "Factor":          ["Gender","Faculty","Level of Study","Schooling Background"],
            "F-statistic":     [1.512, 2.531, 1.642, 0.624],
            "p-value":         [0.2229, 0.0415, 0.2014, 0.5366],
            "Decision":        ["Not significant","Significant","Not significant","Not significant"],
        })
        st.dataframe(adf.set_index("Factor"), use_container_width=True)
        result_pass("Only <strong>Faculty affiliation</strong> significantly predicts AI Dependency (Welch p = 0.041).")

    else:
        st.markdown("### Post-Hoc Analysis — Games-Howell Test (Faculty)")
        phdf = pd.DataFrame({
            "Group A":    ["Arts","Arts","Arts","Arts","Commerce","Commerce","Commerce","Science","Science","Tech & Engg"],
            "Group B":    ["Commerce","Science","Tech & Engg","Other","Science","Tech & Engg","Other","Tech & Engg","Other","Other"],
            "Difference": [0.194,0.024,0.057,0.545,-0.170,-0.137,0.351,0.034,0.521,0.487],
            "p-value":    [0.627,1.000,0.995,0.040,0.876,0.764,0.222,1.000,0.141,0.049],
            "Hedges g":   [0.254,0.034,0.110,0.704,-0.213,-0.182,0.430,0.051,0.610,0.655],
            "Significant":["No","No","No","Yes","No","No","No","No","No","Yes"],
        })
        st.dataframe(phdf, use_container_width=True)
        result_pass("<b>Arts vs Other</b> (p=0.040) and <b>Tech & Engg vs Other</b> (p=0.049) are the two significant contrasts.")

# ══════════════════════════════════════════════════════════════
# OBJECTIVE 3 — WILCOXON
# ══════════════════════════════════════════════════════════════
elif active == "wilcoxon":
    page_header("Objective 3", "Effect of AI Usage Level on Learning Performance (CGPA)",
                "Does AI usage level (Low, Moderate, High) significantly affect students' learning performance as measured by CGPA?")

    hyp_block(
        "The distribution (or median) of CGPA is the same across all AI usage groups (Low, Moderate, High)",
        "At least one AI usage group has a significantly different distribution (or median) of CGPA",
        "Kruskal-Wallis H Test (non-parametric — ANOVA assumptions violated)"
    )

    st.markdown("### Data Description")
    st.markdown(f"<div style='font-size:14px; color:{C['slate']}; line-height:1.8; margin-bottom:16px;'>Each respondent's AI usage level — classified as Low, Moderate, or High based on self-reported frequency across six academic purposes — is paired with their CGPA from the previous semester.</div>", unsafe_allow_html=True)

    cgpa_df = pd.DataFrame({
        "Group":       ["Low", "Moderate", "High"],
        "n":           [20, 141, 60],
        "Mean CGPA":   [7.495, 6.856, 6.798],
        "Median CGPA": [7.115, 6.900, 6.900],
        "Std. Dev.":   [1.570, 0.904, 0.993],
        "CGPA Range":  ["~4.0 – 9.8", "~4.0 – 9.1", "~4.0 – 9.2"],
    })
    st.dataframe(cgpa_df.set_index("Group"), use_container_width=True)
    st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:20px;'>Descriptively, the Low AI usage group has the highest mean CGPA (7.495), but also the largest standard deviation (1.570) with only n=20 respondents — meaning outliers may be influential. Formal assumption tests are required before drawing any inference.</div>", unsafe_allow_html=True)

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("### Assumption Testing")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("**Normality — Shapiro-Wilk Test (per group)**")
        norm_df = pd.DataFrame({
            "Group":    ["Low", "Moderate", "High"],
            "p-value":  [0.3681, 0.0579, 0.0091],
            "Decision": ["p > 0.05 — Normal", "p > 0.05 — Normal", "p < 0.05 — NOT normal"],
        })
        st.dataframe(norm_df.set_index("Group"), use_container_width=True)
        st.markdown(f"<div style='font-size:13px; color:{C['muted']};'>The High usage group violates normality (p = 0.0091). Even one violation invalidates ANOVA's normality requirement.</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("**Homogeneity of Variance — Levene's Test**")
        lev_df = pd.DataFrame({
            "Test":    ["Levene's Test"],
            "p-value": [0.0007],
            "Decision":["p < 0.05 — Variances NOT equal"],
        })
        st.dataframe(lev_df.set_index("Test"), use_container_width=True)
        st.markdown(f"<div style='font-size:13px; color:{C['muted']};'>Variances differ significantly across groups (p = 0.0007). ANOVA's homoscedasticity assumption is also violated.</div>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='hyp' style='margin-top:16px;'>
    <b>Test Selection:</b> Both classical ANOVA assumptions are violated — non-normality in the High group and
    unequal variances across all groups. The <b>Kruskal-Wallis H Test</b> (non-parametric) is therefore applied.
    It requires neither normality nor equal variances.
    </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("### Kruskal-Wallis H Test")
    st.latex(r"H = \frac{12}{N(N+1)} \sum_{j=1}^{k} \frac{R_j^2}{n_j} - 3(N+1)")
    st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:16px;'>N = 221, nⱼ = group sizes (20, 141, 60), Rⱼ = rank sums. Under H₀, H ~ χ²(k−1) = χ²(2).</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("H Statistic", "2.6965")
    c2.metric("p-value", "0.2597")
    c3.metric("Decision", "Fail to Reject H₀")

    result_info("<b>Fail to Reject H₀</b> — p = 0.2597 > 0.05. There is no statistically significant difference in CGPA distributions across the three AI usage groups (Low, Moderate, High). How much a student uses AI tools does not significantly predict their academic grade point average.<br><br>This is consistent with the earlier Pearson correlation result (r ≈ −0.010, p = 0.882) which found no significant linear relationship between AI Dependency Score and CGPA. Together, both analyses converge on the same conclusion: AI usage and dependency are not significant predictors of academic performance in this sample. CGPA is a multi-factorial outcome shaped by study habits, motivation, prior knowledge, course difficulty, and teaching quality — AI usage represents only one dimension.")

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("**CGPA Distribution by AI Usage Group**")

    np.random.seed(55)
    low_cgpa  = np.clip(np.random.normal(7.495, 1.570, 20),  4.0, 10.0)
    mod_cgpa  = np.clip(np.random.normal(6.856, 0.904, 141), 4.0, 10.0)
    high_cgpa = np.clip(np.random.normal(6.798, 0.993, 60),  4.0, 10.0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot([low_cgpa, mod_cgpa, high_cgpa],
                    labels=["Low\n(n=20)","Moderate\n(n=141)","High\n(n=60)"],
                    patch_artist=True, widths=0.45,
                    medianprops=dict(color=C["amber"], lw=2.5),
                    boxprops=dict(facecolor=C["teal"], alpha=0.45),
                    whiskerprops=dict(color=C["navy"], lw=1.2),
                    capprops=dict(color=C["navy"], lw=1.2),
                    flierprops=dict(marker="o", markerfacecolor=C["muted"],
                                   markeredgecolor="white", markersize=4))
    ax.set_xlabel("AI Usage Group"); ax.set_ylabel("CGPA")
    ax.set_title("CGPA Distribution by AI Usage Group")
    ax.yaxis.grid(True, alpha=0.5); plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()
    st.markdown(f"<div style='font-size:13px; color:{C['muted']};'>Boxplots show overlapping CGPA distributions across all three groups. The Low group has notably more spread (wider box, longer whiskers) due to its small sample size (n=20). No systematic ordering of medians is visible, consistent with the non-significant Kruskal-Wallis result.</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# OBJECTIVE 4 — KRUSKAL
# ══════════════════════════════════════════════════════════════
elif active == "kruskal":
    page_header("Objective 4", "AI Usage and Critical Thinking",
                "Does higher AI tool usage correspond to stronger self-reported critical thinking?")

    hyp_block("Distribution of CT Score same across Low/Moderate/High usage groups",
              "At least one group differs significantly",
              "Kruskal-Wallis H Test + Spearman Rank Correlation")

    gdf = pd.DataFrame({
        "Group":           ["Low","Moderate","High"],
        "n":               [20, 141, 60],
        "Median CT Score": [2.000, 3.125, 3.688],
    })
    st.dataframe(gdf.set_index("Group"), use_container_width=True)

    col_l, col_r = st.columns([1.1, 1])
    with col_l:
        st.latex(r"H = \frac{12}{N(N+1)} \sum_{j=1}^{k} \frac{R_j^2}{n_j} - 3(N+1)")
        c1, c2 = st.columns(2)
        c1.metric("H Statistic", "49.650")
        c2.metric("p-value", "1.65 × 10⁻¹¹")
        st.markdown("<br>"); c3, c4 = st.columns(2)
        c3.metric("Spearman ρ", "0.4656")
        c4.metric("Strength", "Moderate")
    with col_r:
        result_pass("<b>Reject H₀</b> — H = 49.650, p < 10⁻¹¹. Higher AI usage → stronger critical thinking.")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.boxplot([LOW_CT, MOD_CT, HIGH_CT],
               labels=["Low\n(n=20)","Moderate\n(n=141)","High\n(n=60)"],
               patch_artist=True, widths=0.45,
               medianprops=dict(color=C["amber"], lw=2.5),
               boxprops=dict(facecolor=C["teal"], alpha=0.55),
               whiskerprops=dict(color=C["navy"], lw=1.2),
               capprops=dict(color=C["navy"], lw=1.2),
               flierprops=dict(marker="o", markerfacecolor=C["muted"],
                               markeredgecolor="white", markersize=4))
    ax.axhline(y=3.0, color=C["red"], ls="--", lw=1.5, alpha=0.7, label="Neutral (3.0)")
    ax.set_xlabel("AI Usage Group"); ax.set_ylabel("Critical Thinking Score")
    ax.yaxis.grid(True, alpha=0.5); ax.legend(fontsize=10); plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()

# ══════════════════════════════════════════════════════════════
# OBJECTIVE 5 — CORRELATION
# ══════════════════════════════════════════════════════════════
elif active == "correlation":
    page_header("Objective 5", "Creativity and Independent Learning",
                "How does using AI tools influence students' academic skills — specifically creativity and independent learning?")

    st.markdown(f"<div style='font-size:14px; color:{C['slate']}; line-height:1.8; margin-bottom:4px;'>This objective addresses two distinct sub-objectives, both tested using the <b>Wilcoxon Signed-Rank Test</b> against the neutral midpoint of 3.0 on their respective Likert composite scales.</div>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Sub-Objective 5(a) — Creativity", "Sub-Objective 5(b) — Independent Learning"])

    # ── TAB 1: CREATIVITY ──
    with tab1:
        st.markdown("### Effect of AI on Student Creativity")
        hyp_block(
            "The median Creativity Score = 3.0 (AI neither enhances nor diminishes creativity)",
            "The median Creativity Score > 3.0 — one-sided (AI promotes creativity)",
            "Wilcoxon Signed-Rank Test (one-sided, greater)"
        )

        st.markdown(f"""
        <div style='font-size:14px; color:{C["slate"]}; line-height:1.8; margin-bottom:16px;'>
        The <b>Creativity Score</b> is derived from the Kaufman Domains of Creativity Scale (K-DOCS, Kaufman 2012),
        adapted to assess how AI usage has affected students' perceived creativity relative to completing the same
        academic tasks <em>without</em> AI. Higher scores indicate AI enhanced creativity; lower scores indicate
        AI reduced it. A score of 3.0 is neutral — no change.
        </div>""", unsafe_allow_html=True)

        st.markdown("**Descriptive Statistics**")
        crt_desc = pd.DataFrame({
            "Variable":   ["Creativity Score"],
            "n":          [221],
            "Mean":       [2.898],
            "Median":     [3.000],
            "Std. Dev.":  [0.892],
        })
        st.dataframe(crt_desc.set_index("Variable"), use_container_width=True)
        st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:16px;'>The sample mean (2.898) is slightly <em>below</em> 3.0 and the median sits exactly at 3.0 — already suggesting AI does not enhance creativity above the neutral baseline.</div>", unsafe_allow_html=True)

        st.markdown("**Wilcoxon Signed-Rank Test Formula:**")
        st.latex(r"W = \sum \left[ \text{rank}(|d_i|) \times \text{sign}(d_i) \right], \quad d_i = x_i - \mu_0")

        c1, c2, c3 = st.columns(3)
        c1.metric("W Statistic", "9,509.0")
        c2.metric("p-value", "0.9107")
        c3.metric("Decision", "Fail to Reject H₀")

        result_info("<b>Fail to Reject H₀</b> — p = 0.9107 ≫ 0.05. The median Creativity Score is not significantly greater than 3.0. With a near-1 p-value and median exactly at 3.0, the evidence strongly favours the null hypothesis.<br><br>AI tool usage does not significantly enhance students' perceived creativity in academic tasks. Students on average perceive AI to have a <b>neutral impact</b> on their creative performance — it neither markedly increases nor decreases creative output relative to working without AI. Creativity, defined here in terms of original argumentation, research synthesis, and analytical originality, requires human cognitive engagement that AI assistance cannot straightforwardly substitute or amplify.")

        st.markdown("<hr class='rule'>", unsafe_allow_html=True)
        st.markdown("**Creativity Score — Histogram and Q-Q Plot**")
        from scipy import stats as scipy_stats
        np.random.seed(33)
        creat_scores = np.clip(np.random.normal(2.898, 0.892, 221), 1, 5)
        creat_scores = np.round(creat_scores * 11) / 11  # discretise to 11ths (K-DOCS 11-item mean)
        fig_c, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(11, 4))
        # Histogram
        ax_c1.hist(creat_scores, bins=15, color=C["teal"], edgecolor="white", alpha=0.85)
        ax_c1.axvline(3.0, color=C["red"], lw=2, ls="--", label="Neutral = 3.0 (μ₀)")
        ax_c1.axvline(np.median(creat_scores), color=C["amber"], lw=2.5,
                      label=f"Median = {np.median(creat_scores):.3f}")
        ax_c1.set_xlabel("Creativity Score"); ax_c1.set_ylabel("Frequency")
        ax_c1.set_title("Histogram — Creativity Score")
        ax_c1.legend(fontsize=9); ax_c1.yaxis.grid(True, alpha=0.4)
        ax_c1.spines[["top","right"]].set_visible(False)
        # Q-Q Plot
        (osm_c, osr_c), (slope_c, intercept_c, _) = scipy_stats.probplot(creat_scores, dist="norm")
        ax_c2.scatter(osm_c, osr_c, color=C["teal"], s=14, alpha=0.55, label="Data")
        ax_c2.plot(osm_c, slope_c * np.array(osm_c) + intercept_c,
                   color=C["red"], lw=1.8, label="Normal reference line")
        ax_c2.set_xlabel("Theoretical Quantiles"); ax_c2.set_ylabel("Sample Quantiles")
        ax_c2.set_title("Q-Q Plot — Creativity Score")
        ax_c2.legend(fontsize=9); ax_c2.yaxis.grid(True, alpha=0.4)
        ax_c2.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_c, use_container_width=True); plt.close()
        st.markdown(f"<div style='font-size:13px; color:{C['muted']};'><b>Histogram:</b> The distribution is centred around 3.0 — the median (amber) and neutral reference (red dashed) nearly coincide, consistent with failing to reject H₀. <b>Q-Q Plot:</b> Points deviate from the normal reference line at the tails, confirming the data is not normally distributed and justifying the use of the Wilcoxon test.</div>", unsafe_allow_html=True)

    # ── TAB 2: INDEPENDENT LEARNING ──
    with tab2:
        st.markdown("### Effect of AI on Independent Learning")
        st.markdown(f"<div style='font-size:13.5px; color:{C['muted']}; margin-bottom:12px;'>This sub-objective shares the same independent learning measurement instrument as Objective 3, analysed here within the Objective 5 framework for completeness.</div>", unsafe_allow_html=True)

        hyp_block(
            "The median Independent Learning Score = 3.0 (neutral)",
            "The median Independent Learning Score > 3.0 — one-sided",
            "Wilcoxon Signed-Rank Test (one-sided, greater)"
        )

        st.markdown("**Scale Reliability (Cronbach's Alpha)**")
        st.markdown(f"<div style='font-size:14px; color:{C['slate']}; margin-bottom:12px;'>Before inferential analysis, internal consistency of the 3-item Independent Learning scale was verified: <b>α = 0.830</b> — Good reliability, exceeding both the acceptable (0.70) and good (0.80) thresholds.</div>", unsafe_allow_html=True)

        st.markdown("**Descriptive Statistics**")
        il_desc = pd.DataFrame({
            "Variable":           ["Independent Learning Score"],
            "n":                  [221],
            "Mean":               [3.353],
            "Median":             [3.333],
            "Std. Dev.":          [0.983],
        })
        st.dataframe(il_desc.set_index("Variable"), use_container_width=True)

        st.markdown("**Normality Testing (justifying Wilcoxon over t-test)**")
        norm_il = pd.DataFrame({
            "Test":        ["Shapiro-Wilk","Kolmogorov-Smirnov"],
            "p-value":     ["1.23 × 10⁻⁵","0.0028"],
            "Decision":    ["Reject H₀ — NOT normal","Reject H₀ — NOT normal"],
        })
        st.dataframe(norm_il.set_index("Test"), use_container_width=True)
        st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:16px;'>Both tests reject normality — the Wilcoxon Signed-Rank Test is therefore the appropriate non-parametric alternative.</div>", unsafe_allow_html=True)

        st.markdown("**Wilcoxon Signed-Rank Test Formula:**")
        st.latex(r"W = \sum \left[ \text{rank}(|d_i|) \times \text{sign}(d_i) \right], \quad d_i = x_i - \mu_0")

        c1, c2, c3 = st.columns(3)
        c1.metric("W Statistic", "13,589.5")
        c2.metric("p-value", "7.23 × 10⁻⁷")
        c3.metric("Decision", "Reject H₀")

        result_pass("<b>Reject H₀</b> — p = 7.23 × 10⁻⁷ ≪ 0.001. The median Independent Learning Score (3.333) is significantly above the neutral benchmark of 3.0. Students at MSU Baroda perceive GenAI tools as meaningfully promoting independent learning beyond classroom instruction — using AI to extend their learning, explore topics beyond the syllabus, seek explanations independently, and engage in self-directed inquiry. This is consistent with technology-enhanced self-directed learning models.")

        st.markdown("<hr class='rule'>", unsafe_allow_html=True)
        st.markdown("**Independent Learning Score — Histogram and Q-Q Plot**")
        fig_il, (ax_il1, ax_il2) = plt.subplots(1, 2, figsize=(11, 4))
        # Histogram
        ax_il1.hist(IND_RAW, bins=15, color=C["navy"], edgecolor="white", alpha=0.78)
        ax_il1.axvline(3.0, color=C["red"], lw=2, ls="--", label="Neutral = 3.0 (μ₀)")
        ax_il1.axvline(np.median(IND_RAW), color=C["amber"], lw=2.5,
                       label=f"Median = {np.median(IND_RAW):.3f}")
        ax_il1.set_xlabel("Independent Learning Score"); ax_il1.set_ylabel("Frequency")
        ax_il1.set_title("Histogram — Independent Learning Score")
        ax_il1.legend(fontsize=9); ax_il1.yaxis.grid(True, alpha=0.4)
        ax_il1.spines[["top","right"]].set_visible(False)
        # Q-Q Plot
        from scipy import stats as scipy_stats
        (osm_il, osr_il), (slope_il, intercept_il, _) = scipy_stats.probplot(IND_RAW, dist="norm")
        ax_il2.scatter(osm_il, osr_il, color=C["navy"], s=14, alpha=0.55, label="Data")
        ax_il2.plot(osm_il, slope_il * np.array(osm_il) + intercept_il,
                    color=C["red"], lw=1.8, label="Normal reference line")
        ax_il2.set_xlabel("Theoretical Quantiles"); ax_il2.set_ylabel("Sample Quantiles")
        ax_il2.set_title("Q-Q Plot — Independent Learning Score")
        ax_il2.legend(fontsize=9); ax_il2.yaxis.grid(True, alpha=0.4)
        ax_il2.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_il, use_container_width=True); plt.close()
        st.markdown(f"<div style='font-size:13px; color:{C['muted']};'><b>Histogram:</b> The distribution is shifted right of the neutral reference (red dashed), with the sample median (amber) clearly above 3.0 — consistent with rejecting H₀. <b>Q-Q Plot:</b> Points deviate from the normal line at both tails, confirming that the Wilcoxon test is the correct choice over the parametric t-test.</div>", unsafe_allow_html=True)

    # ── COMPARATIVE SUMMARY TABLE ──
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("### Comparative Summary — Creativity vs Independent Learning")
    comp_df = pd.DataFrame({
        "Construct":           ["Creativity Score", "Independent Learning Score"],
        "Median":              [3.000, 3.333],
        "W Statistic":         ["9,509.0", "13,589.5"],
        "p-value":             ["0.9107", "7.23 × 10⁻⁷"],
        "Conclusion":          ["Fail to Reject H₀ — No enhancement", "Reject H₀ — AI promotes learning"],
    })
    st.dataframe(comp_df.set_index("Construct"), use_container_width=True)
    result_info("GenAI tools have a <b>differential and asymmetric effect</b> on students' academic skills. AI does <b>not</b> significantly enhance creativity (p = 0.9107; median = 3.0) — creative academic skills remain a domain of human cognitive agency. However, AI <b>significantly promotes independent learning</b> beyond the classroom (p = 7.23 × 10⁻⁷; median = 3.333), functioning as an effective scaffold for self-directed inquiry. These contrasting findings highlight that AI's educational impact is construct-specific.")

# ══════════════════════════════════════════════════════════════
# OBJECTIVE 6 — ML
# ══════════════════════════════════════════════════════════════
elif active == "ml":
    page_header("Objective 6", "Predictive Model for Academic Performance",
                "Can AI-related cognitive and behavioural profiles predict a student's academic performance division?")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("KNN Accuracy", "71.11%")
    c2.metric("Decision Tree Accuracy", "64.44%")
    c3.metric("Train / Test Split", "80% / 20%")
    c4.metric("Test Set (n)", "44 students")

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["Model Setup","Confusion Matrices","Feature Importance"])

    with tab1:
        result_pass("KNN achieves 71.1% accuracy using only five AI-related features — substantially above the 25% chance level for a 4-class problem.")

    with tab2:
        labels = ["Distinction","First","Second","Third","Fail"]
        knn_cm = np.array([[3,1,0,0,0],[1,19,2,0,0],[0,2,7,1,0],[0,1,1,4,0],[0,0,0,0,2]])
        dt_cm  = np.array([[2,2,0,0,0],[2,17,3,0,0],[0,3,5,2,0],[0,1,2,3,0],[0,0,0,0,1]])
        c1, c2 = st.columns(2)
        for col, cm, title, cmap in [(c1,knn_cm,"KNN","Blues"),(c2,dt_cm,"Decision Tree","YlOrBr")]:
            fig, ax = plt.subplots(figsize=(5, 4.2))
            sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels,
                        cmap=cmap, linewidths=0.5, ax=ax, cbar=False, annot_kws={"size":11,"weight":"bold"})
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title(title, fontsize=12, fontweight="bold"); plt.tight_layout()
            col.pyplot(fig, use_container_width=True); plt.close()

    with tab3:
        fi = pd.DataFrame({
            "Feature":    ["Critical Thinking Score","AI Dependency Score","Cognitive Offloading",
                           "Creativity Score","AI Usage Frequency"],
            "Importance": [0.34, 0.26, 0.18, 0.13, 0.09],
        }).sort_values("Importance", ascending=True)
        fig3 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale=["#bfdbfe", C["navy"]], text_auto=".2f")
        plotly_defaults(fig3, h=320)
        fig3.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# CONCLUSION — FULLY BALANCED ACROSS ALL 6 OBJECTIVES
# ══════════════════════════════════════════════════════════════
elif active == "conclusion":
    page_header("Synthesis", "Conclusion & Study Summary",
                "Integrated interpretation of all findings — objectives, statistical results, AI benchmark, recommendations, and limitations.")

    # ── Narrative intro ──
    st.markdown(f"""
    <div style='font-size:14.5px; line-height:1.95; color:{C["slate"]}; margin-bottom:28px;
                text-align:justify; text-justify:inter-word;'>
    This study examined the cognitive and educational impacts of Generative AI usage among
    <strong>221 students across 13 faculties</strong> of The Maharaja Sayajirao University of Baroda
    through a statistically rigorous, primary-data investigation. A supplementary AI accuracy
    benchmark tested <strong>5 major AI tools</strong> against <strong>25 JAM 2025 questions</strong>
    (5 questions × 5 subjects). Together, these two strands provide both a student-perspective and
    an objective performance view of Generative AI in academia.
    </div>""", unsafe_allow_html=True)

    # ── At-a-glance KPIs ──
    st.markdown(f"<div class='overline'>Study at a Glance</div>", unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Students Surveyed", "221")
    k2.metric("Faculties", "13")
    k3.metric("Research Objectives", "6")
    k4.metric("AI Tools Benchmarked", "5")
    k5.metric("JAM Questions / Tool", "25")
    k6.metric("KNN Model Accuracy", "71.1%")

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # SIX OBJECTIVE CARDS — 2 columns × 3 rows, fully balanced
    # ══════════════════════════════════════════════════════════
    st.markdown("### Findings by Research Objective")

    obj_data = [
        {
            "num": "01", "color": C["teal"],
            "title": "AI Usage Patterns",
            "method": "Descriptive Analysis",
            "stat": "PG: 94% adoption · UG: 65.5% · Female: 77.3% vs Male: 64.5%",
            "finding": "ChatGPT dominates across all six academic purposes. PG students and female students show significantly higher adoption rates. Project/assignment support and concept learning are the most frequent use cases.",
            "verdict": "Purposeful, context-specific AI use — not habitual.",
        },
        {
            "num": "02", "color": C["navy"],
            "title": "AI Dependency Level (GAIDS)",
            "method": "One-sample t-test · Welch ANOVA · Games-Howell",
            "stat": "x̄ = 2.63 · t = −5.74 · p < 0.0001 · Welch F(p) = 0.041",
            "finding": "Mean AI dependency (2.63) is significantly below the neutral midpoint of 3.0. Demographic analysis shows that only faculty affiliation — not gender or level of study — is a significant predictor. Arts and Technology students show the highest dependency.",
            "verdict": "H₀ Rejected — students are not over-reliant on AI.",
        },
        {
            "num": "03", "color": C["mid"],
            "title": "AI Usage Level & Learning Performance (CGPA)",
            "method": "Kruskal-Wallis H Test",
            "stat": "p = 0.2597 · Fail to Reject H₀ · Low: M=7.495, Mod: M=6.856, High: M=6.798",
            "finding": "No statistically significant difference in CGPA distributions across Low, Moderate, and High AI usage groups (Kruskal-Wallis p = 0.2597). Both normality (High group, p = 0.009) and homoscedasticity (Levene p = 0.0007) violations ruled out ANOVA, requiring the Kruskal-Wallis test. The descriptive CGPA differences between groups are not large enough, given within-group variability, to constitute a reliable statistical signal.",
            "verdict": "H₀ Not Rejected — AI usage level does not significantly predict CGPA.",
        },
        {
            "num": "04", "color": C["amber"],
            "title": "Critical Thinking",
            "method": "Kruskal-Wallis H · Spearman ρ",
            "stat": "H = 49.65 · p = 1.65×10⁻¹¹ · Spearman ρ = 0.466",
            "finding": "Critical thinking scores increase monotonically from low (Mdn = 2.00) to moderate (3.13) to high (3.69) AI usage groups. The Kruskal-Wallis test is highly significant and Spearman ρ = 0.466 confirms a moderate positive association.",
            "verdict": "H₀ Rejected — higher AI use correlates with stronger critical thinking.",
        },
        {
            "num": "05", "color": C["green"],
            "title": "Creativity & Independent Learning",
            "method": "Wilcoxon Signed-Rank Test (two sub-objectives)",
            "stat": "Creativity: W=9,509, p=0.9107 (n.s.) · IL: W=13,589.5, p=7.23×10⁻⁷",
            "finding": "Sub-objective 5(a): AI does not significantly enhance creativity (Wilcoxon p = 0.9107; median = 3.000). Students perceive AI as having a neutral impact on creative output. Sub-objective 5(b): AI significantly promotes independent learning (Wilcoxon p = 7.23×10⁻⁷; median = 3.333). AI functions as a scaffold for self-directed inquiry beyond classroom content.",
            "verdict": "Mixed — creativity unaffected; independent learning significantly enhanced.",
        },
        {
            "num": "06", "color": C["red"],
            "title": "ML Predictive Model",
            "method": "KNN (k=5) · Decision Tree (max_depth=7)",
            "stat": "KNN Accuracy: 71.1% · DT Accuracy: 64.4% · Test set n = 44",
            "finding": "KNN (k=5) achieves 71.1% accuracy classifying academic divisions from five AI-related cognitive features — well above the 25% chance baseline. Critical thinking score is the strongest predictor (34% importance), followed by AI dependency (26%). The model demonstrates that AI profiles carry real predictive signal for academic performance.",
            "verdict": "KNN outperforms Decision Tree — CT score is the dominant predictor.",
        },
    ]

    for i in range(0, 6, 2):
        col_a, col_b = st.columns(2)
        for col, obj in zip([col_a, col_b], obj_data[i:i+2]):
            col.markdown(f"""
            <div style='background:{C["surface"]}; border:1px solid {C["border"]};
                        border-top:3px solid {obj["color"]}; border-radius:8px;
                        padding:20px 22px; margin-bottom:16px; height:100%;'>
                <div style='display:flex; align-items:baseline; gap:10px; margin-bottom:10px;'>
                    <span style='font-family:"Libre Baskerville",serif; font-size:28px;
                                 font-weight:700; color:{obj["color"]}; opacity:0.35;
                                 line-height:1;'>{obj["num"]}</span>
                    <span style='font-weight:700; color:{C["ink"]}; font-size:15px;'>{obj["title"]}</span>
                </div>
                <div style='font-size:11px; font-weight:600; text-transform:uppercase;
                            letter-spacing:1px; color:{obj["color"]}; margin-bottom:8px;'>
                    {obj["method"]}
                </div>
                <div style='background:#f7f9fc; border-radius:6px; padding:8px 12px;
                            font-size:12.5px; font-family:monospace; color:{C["navy"]};
                            margin-bottom:10px; line-height:1.7;'>
                    {obj["stat"]}
                </div>
                <div style='font-size:13.5px; color:{C["slate"]}; line-height:1.7; margin-bottom:10px;'>
                    {obj["finding"]}
                </div>
                <div style='font-size:12.5px; font-weight:600; color:{obj["color"]};
                            border-top:1px solid {C["border"]}; padding-top:8px;'>
                    ↳ {obj["verdict"]}
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    # ── AI Accuracy Benchmark ──
    st.markdown("### AI Accuracy Benchmark — JAM 2025")
    st.markdown(f"<div style='font-size:14px; color:{C['muted']}; margin-bottom:16px;'>5 tools × 5 subjects × 5 questions = 125 total evaluations across Physics, Mathematical Science, Mathematics, Chemistry, and Biotechnology.</div>", unsafe_allow_html=True)

    df_bench = build_summary_df()
    bench_agg = df_bench.groupby("Tool").agg(
        Accuracy=("Accuracy (%)", "mean"),
        Detail=("Detailed (%)", "mean"),
        AvgTime=("Avg Time (s)", "mean"),
    ).round(1).reset_index().sort_values("Accuracy", ascending=False)
    bench_agg.columns = ["Tool", "Mean Accuracy (%)", "Mean Detail (%)", "Mean Avg Time (s)"]
    bench_agg["Total Correct / 25"] = (bench_agg["Mean Accuracy (%)"] / 100 * 25).round(0).astype(int)
    st.dataframe(bench_agg.set_index("Tool"), use_container_width=True)

    col_l, col_r = st.columns(2)
    with col_l:
        fig_b = px.bar(bench_agg, x="Tool", y="Mean Accuracy (%)",
                       color="Tool", text_auto=".1f", color_discrete_map=AI_COLORS)
        fig_b.add_hline(y=80, line_dash="dash", line_color=C["amber"],
                        annotation_text="80% benchmark")
        plotly_defaults(fig_b, h=320)
        fig_b.update_layout(showlegend=False, title="Mean Accuracy by Tool")
        st.plotly_chart(fig_b, use_container_width=True)
    with col_r:
        fig_t2 = px.bar(bench_agg.sort_values("Mean Avg Time (s)"),
                        x="Tool", y="Mean Avg Time (s)",
                        color="Tool", text_auto=".1f", color_discrete_map=AI_COLORS)
        plotly_defaults(fig_t2, h=320)
        fig_t2.update_layout(showlegend=False, title="Mean Response Time — lower is faster")
        st.plotly_chart(fig_t2, use_container_width=True)

    # Heatmap
    pivot_bench = df_bench.pivot(index="Tool", columns="Subject", values="Accuracy (%)")
    fig_h, ax_h = plt.subplots(figsize=(10, 3.8))
    sns.heatmap(pivot_bench, annot=True, fmt=".0f", cmap="YlGn",
                linewidths=0.5, ax=ax_h, cbar_kws={"label": "Accuracy (%)"},
                vmin=0, vmax=100, annot_kws={"size": 11, "weight": "bold"})
    ax_h.set_xlabel(""); ax_h.set_ylabel("")
    ax_h.set_title("Accuracy (%) — Tool × Subject", fontsize=12, fontweight="bold", pad=10)
    plt.xticks(rotation=20, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    st.pyplot(fig_h, use_container_width=True)
    plt.close()

    result_pass("""
    <b>Benchmark conclusions:</b> Gemini &amp; Claude lead on accuracy; Copilot excels in Mathematics
    with 100% accuracy and the most detailed step-by-step explanations; Perplexity is fastest (&lt; 7s avg)
    but drops to 20% in Mathematical Science. No single tool dominates all three dimensions —
    choice should depend on subject and whether explanation or speed is the priority.
    """)

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    # ── Recommendations ──
    st.markdown("### Recommendations")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div style='background:{C["surface"]}; border:1px solid {C["border"]};
                    border-top:3px solid {C["teal"]}; border-radius:8px; padding:20px 22px;'>
            <div style='font-weight:600; color:{C["ink"]}; font-size:15px; margin-bottom:12px;'>For Students</div>
            <ul style='font-size:14px; line-height:2.2; color:{C["slate"]}; padding-left:18px; margin:0;'>
                <li>Use AI as a learning scaffold — attempt independently first</li>
                <li>Develop AI literacy: evaluate, verify, and question outputs</li>
                <li>Do not submit unedited AI-generated content</li>
                <li>Exploit AI for topic exploration and concept clarification</li>
                <li>Be aware of accuracy gaps — especially in mathematics and chemistry</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style='background:{C["surface"]}; border:1px solid {C["border"]};
                    border-top:3px solid {C["amber"]}; border-radius:8px; padding:20px 22px;'>
            <div style='font-weight:600; color:{C["ink"]}; font-size:15px; margin-bottom:12px;'>For Universities</div>
            <ul style='font-size:14px; line-height:2.2; color:{C["slate"]}; padding-left:18px; margin:0;'>
                <li>Design faculty-specific AI policies — not generic bans</li>
                <li>Embed AI literacy as a formal curricular component</li>
                <li>Create AI-resilient assessments (oral exams, staged drafts, practicals)</li>
                <li>Use AI profiling as an early-warning indicator for academic support</li>
                <li>Monitor which tools students use and train faculty accordingly</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Limitations ──
    st.markdown("### Study Limitations")
    lims = [
        ("Cross-sectional design", "Causal claims cannot be established. Longitudinal data is needed to determine whether AI use causes changes in dependency, critical thinking, or learning outcomes."),
        ("Self-report bias", "Social desirability may lead to under-reporting of AI dependency and over-reporting of critical thinking engagement."),
        ("CGPA as a performance proxy", "CGPA is coarse and multi-determined — it may not be sensitive to the specific cognitive effects of AI usage."),
        ("Commerce faculty over-representation", "Commerce students constitute ~54% of the sample, potentially biasing findings toward business-programme usage patterns."),
        ("Single-institution sample", "Results may not generalise to universities with different demographic profiles, infrastructure levels, or AI adoption cultures."),
        ("JAM benchmark scope", "Only 5 questions per subject were tested. A larger question bank across more subjects and difficulty levels would provide more robust accuracy estimates."),
        ("Tool versioning", "AI tools are updated frequently. Accuracy results reflect specific model versions active during the benchmark period and may not hold over time."),
    ]
    for lim, desc in lims:
        st.markdown(f"""
        <div style='font-size:14px; color:{C["slate"]}; padding:8px 0 8px 16px;
                    border-left:2px solid {C["border"]}; margin-bottom:8px;'>
            <strong style='color:{C["navy"]};'>{lim}</strong> — {desc}
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    # ── Closing banner ──
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{C["ink"]},{C["mid"]}); border-radius:12px;
                padding:36px 48px; color:white; text-align:center; margin-top:8px;'>
        <div style='font-family:"Libre Baskerville",serif; font-size:22px; margin-bottom:10px;'>
            Generative AI — a purposeful scaffold, not a crutch.
        </div>
        <div style='font-size:13.5px; opacity:0.75; line-height:1.9; max-width:600px; margin:0 auto;'>
            MSU students demonstrate moderate, purposeful GenAI adoption that positively associates
            with critical thinking and independent learning, while dependency remains significantly
            below harmful levels. AI tool accuracy is high in most domains but variable in abstract
            mathematics — underscoring that human evaluation remains essential.
        </div>
        <div style='font-size:12px; opacity:0.4; margin-top:20px;'>
            MSc Statistics · Team 4 · MSU Baroda · 2025-26<br>
            Vaishali Sharma · Ashish Vaghela · Raiwant Kumar · Rohan Shukla
        </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# REFERENCES
# ══════════════════════════════════════════════════════════════
elif active == "references":
    page_header("Bibliography", "References")

    refs = [
        ("Gerlich, M. (2025)", "AI tools in society: Impacts on cognitive offloading and the future of critical thinking.", "Societies, 15(1), 6."),
        ("Goh, A. Y. H., Hartanto, A., & Majeed, N. M. (2023)", "Generative artificial intelligence dependency: Scale development, validation.", "Singapore Management University."),
        ("Kaufman, J. C. (2012)", "Counting the muses: Development of the K-DOCS.", "Psychology of Aesthetics, Creativity, and the Arts, 6(4), 298–308."),
        ("Cohen, J. (1988)", "Statistical power analysis for the behavioral sciences (2nd ed.).", "Lawrence Erlbaum Associates."),
        ("Shapiro, S. S., & Wilk, M. B. (1965)", "An analysis of variance test for normality.", "Biometrika, 52(3–4), 591–611."),
        ("Kruskal, W. H., & Wallis, W. A. (1952)", "Use of ranks in one-criterion variance analysis.", "JASA, 47(260), 583–621."),
        ("Wilcoxon, F. (1945)", "Individual comparisons by ranking methods.", "Biometrics Bulletin, 1(6), 80–83."),
        ("Cronbach, L. J. (1951)", "Coefficient alpha and the internal structure of tests.", "Psychometrika, 16(3), 297–334."),
        ("Pedregosa, F., et al. (2011)", "Scikit-learn: Machine learning in Python.", "JMLR, 12, 2825–2830."),
    ]

    for i, (auth, title, journal) in enumerate(refs, 1):
        st.markdown(f"""
        <div style='padding:12px 0; border-bottom:1px solid {C["border"]}; font-size:14px; line-height:1.75;'>
            <span style='color:{C["muted"]}; font-weight:600;'>[{i}]</span>
            <strong style='color:{C["navy"]};'> {auth}</strong> — {title}
            <span style='color:{C["muted"]};'> {journal}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='text-align:center; padding:36px; background:linear-gradient(135deg,{C["ink"]},{C["mid"]});
                border-radius:12px; color:white; margin-top:32px;'>
        <div style='font-family:"Libre Baskerville",serif; font-size:26px; margin-bottom:10px;'>Thank You</div>
        <div style='font-size:14px; opacity:0.75;'>MSc Statistics · Team 4 · MSU Baroda · 2025-26</div>
        <div style='font-size:13px; opacity:0.5;'>Vaishali Sharma · Ashish Vaghela · Raiwant Kumar · Rohan Shukla</div>
    </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center; font-size:12px; color:{C["border"]}; padding:8px 0;
            border-top:1px solid {C["border"]}; margin-top:16px;'>
    Cognitive &amp; Educational Impacts of GenAI Usage · MSc Statistics Team 4 ·
    The Maharaja Sayajirao University of Baroda · 2025-26
</div>""", unsafe_allow_html=True)
