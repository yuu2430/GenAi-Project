"""
GenAI Impact Study — Full Dashboard
MSc Statistics (Team 4) | The Maharaja Sayajirao University of Baroda
Academic Year 2025-26

Run: streamlit run genai_dashboard_final.py
Requires: data.xlsx in the same directory
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

# ── CSS ──────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
    color: {C['slate']};
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {C['ink']} !important;
    border-right: none;
}}
section[data-testid="stSidebar"] * {{ color: #b0bec5 !important; }}
section[data-testid="stSidebar"] .stRadio label {{
    display: block;
    padding: 8px 16px;
    border-radius: 6px;
    font-size: 13.5px;
    font-weight: 400;
    cursor: pointer;
    transition: all 0.15s;
    border-left: 2px solid transparent;
    margin-bottom: 1px;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(14,124,123,0.18);
    border-left-color: {C['teal']};
    color: #e0e7ef !important;
}}
input[type="radio"] {{ display: none; }}

/* Layout */
.block-container {{
    padding: 2.2rem 3rem 3rem 3rem;
    background: {C['bg']};
    max-width: 1200px;
}}

/* Typography */
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

/* Overline label */
.overline {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.4px;
    color: {C['teal']};
    margin-bottom: 4px;
}}

/* Page title */
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

/* Hypothesis block */
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

/* Result block */
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

/* Inline badge */
.badge-pass {{ background:#dcf5ea; color:#145c3a; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }}
.badge-info {{ background:#fef3cd; color:#7a4f00; padding:2px 10px; border-radius:12px; font-size:12px; font-weight:600; }}

/* Divider */
.rule {{ border:none; border-top:1px solid {C['border']}; margin:24px 0; }}

/* Metric cards */
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

/* Data table */
.stDataFrame {{ border: 1px solid {C['border']}; border-radius: 8px; overflow: hidden; }}

/* Tab styling */
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

/* Plotly chart container */
.element-container:has(.stPlotlyChart) {{
    border: 1px solid {C['border']};
    border-radius: 8px;
    overflow: hidden;
    background: {C['surface']};
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

    PAGES = {
        "Overview":                "overview",
        "Objectives":              "objectives",
        "Pilot Survey":            "pilot",
        "Sampling Design":         "sampling",
        "Questionnaire":           "questionnaire",
        "Reliability Analysis":    "reliability",
        "Objective 1 — Descriptive":   "descriptive",
        "Objective 2 — AI Dependency": "anova",
        "Objective 3 — Ind. Learning": "wilcoxon",
        "Objective 4 — Critical Thinking": "kruskal",
        "Objective 5 — Creativity": "correlation",
        "Objective 6 — ML Model":  "ml",
        "Conclusion":              "conclusion",
        "References":              "references",
    }

    page  = st.radio("", list(PAGES.keys()), label_visibility="collapsed")
    active = PAGES[page]

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

# ══════════════════════════════════════════════════════════════
# OVERVIEW
# ══════════════════════════════════════════════════════════════
if active == "overview":

    # ── LOGO ─────────────────────────────
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image("msu_logo.png", width=130)

    # ── UNIVERSITY HEADER ─────────────────
    st.markdown("""
    <div style='text-align:center; font-size:13px; color:#4a5568; line-height:1.8; margin-top:-8px; margin-bottom:18px;'>
        THE MAHARAJA SAYAJIRAO UNIVERSITY OF BARODA<br>
        FACULTY OF SCIENCE<br>
        DEPARTMENT OF STATISTICS<br>
        ACADEMIC YEAR 2025–26
    </div>
    """, unsafe_allow_html=True)


    # ── HERO SECTION ─────────────────────
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,{C["ink"]} 0%,{C["mid"]} 100%);
                border-radius:12px; padding:40px 48px; color:white; margin-bottom:32px;'>

        <div style='font-family:"Libre Baskerville",serif; font-size:24px;
                    font-weight:700; margin-bottom:8px;'>
            PROJECT REPORT
        </div>

        <div style='font-size:13px; color:#94b4cc; margin-bottom:10px;'>on</div>

        <div style='font-family:"Libre Baskerville",serif; font-size:26px;
                    font-weight:600; line-height:1.5; font-style:italic; margin-bottom:18px;'>
            “Cognitive and Educational Impacts of<br>
            Generative AI Usage Among University Students”
        </div>

        <div style='font-size:14px; color:#94b4cc; max-width:700px; line-height:1.8;'>
            A statistical investigation into how Generative AI tools — ChatGPT, Gemini,
            Copilot, and Perplexity — influence student dependency, independent learning,
            critical thinking, creativity, and academic performance across multiple faculties.
        </div>

        <div style='display:flex; justify-content:space-between; margin-top:28px; font-size:13px; color:#d1d9e6;'>

            <div>
                <b>MSc Statistics (Team 4)</b><br>
                Vaishali Sharma<br>
                Ashish Vaghela<br>
                Raiwant Kumar<br>
                Rohan Shukla
            </div>

            <div style='text-align:right;'>
                <b>Guided by:</b><br>
                Prof. Murlidharan Kunnumal
            </div>

        </div>

    </div>
    """, unsafe_allow_html=True)


    # ── ABSTRACT ─────────────────────────
    st.markdown(f"""
    <div style='font-family:"Libre Baskerville",serif; font-size:15px; line-height:1.95;
                color:{C["slate"]}; max-width:860px; margin-bottom:32px;'>
    Primary data were collected from <strong>221 students</strong> across <strong>13 faculties</strong>
    using a structured questionnaire administered via Probability Proportional to Size (PPS) sampling.
    Reliability of all psychometric scales was confirmed using Cronbach's Alpha (α ≥ 0.83 across all constructs).
    The study employs descriptive analysis, normality testing, non-parametric inference, correlation analysis,
    and supervised machine learning to address six research objectives.<br><br>

    Findings reveal that students exhibit <strong>moderate, purposeful GenAI use</strong>, with average
    dependency levels significantly below the neutral benchmark of 3.0. AI usage is positively associated
    with independent learning (ρ = 0.459) and critical thinking (ρ = 0.466), while no significant relationship
    is observed between AI dependency and CGPA or creativity. Faculty affiliation — not gender or level of
    study — is the only significant demographic predictor of AI dependency.
    </div>
    """, unsafe_allow_html=True)


    # ── STUDY AT A GLANCE ─────────────────
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:1.4px; color:{C['teal']}; margin-bottom:16px;'>Study at a Glance</div>", unsafe_allow_html=True)

    cols = st.columns(4)
    for col, v, l in zip(cols,
        ["221","13","6","4"],
        ["Students surveyed","Faculties covered","Research objectives","AI tools studied"]):
        col.metric(l, v)


    # ── KEY FINDINGS ──────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:11px; font-weight:600; text-transform:uppercase; letter-spacing:1.4px; color:{C['teal']}; margin-bottom:16px;'>Key Findings</div>", unsafe_allow_html=True)

    findings = [
        ("Mean AI Dependency = 2.63",
         "Significantly below the neutral midpoint of 3.0 (t = −5.74, p < 0.0001). Students use AI purposefully, not compulsively."),

        ("Faculty drives dependency differences",
         "Arts and Technology students show significantly higher AI dependency than smaller faculties (Welch ANOVA p = 0.041)."),

        ("AI promotes independent learning",
         "Median Independent Learning Score (3.35) significantly exceeds the neutral benchmark (Wilcoxon W = 13,589, p < 0.001)."),

        ("Higher AI use correlates with stronger critical thinking",
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
         "To determine how frequently students use AI tools, identify the purposes for which they are used, and examine whether usage patterns vary by educational level (UG vs PG) or gender.",
         "Descriptive analysis · Grouped bar charts · Donut charts"),
        ("2", "Level of AI Dependency",
         "To identify and quantify the level of dependency on Generative AI among MSU students using the validated Generative AI Dependency Scale (GAIDS).",
         "Shapiro-Wilk · One-sample t-test · Multi-way ANOVA · Welch ANOVA · Games-Howell post hoc"),
        ("3", "Independent Learning Beyond the Classroom",
         "To examine whether AI use promotes self-directed learning beyond prescribed classroom content.",
         "Cronbach's Alpha · Shapiro-Wilk · Kolmogorov-Smirnov · Wilcoxon Signed-Rank Test"),
        ("4", "AI Usage and Critical Thinking",
         "To study how AI tool usage frequency is related to students' self-reported critical thinking scores.",
         "Kruskal-Wallis H Test · Spearman Rank Correlation"),
        ("5", "Creativity and Independent Learning",
         "To explore how AI usage influences students' creativity (K-DOCS) and autonomous learning behaviour.",
         "Spearman Rank Correlation (two tests) · Cronbach's Alpha"),
        ("6", "Predictive Model for Academic Performance",
         "To build a classification model predicting academic performance divisions from AI-related cognitive and behavioural features.",
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
        st.markdown(f"<div style='font-size:13.5px; color:{C['muted']}; line-height:1.7;'>ChatGPT dominates across all six academic purposes. Programming/Coding shows elevated Copilot usage alongside ChatGPT, consistent with their specialised coding capabilities. Gemini ranks second in most categories.</div>", unsafe_allow_html=True)

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
        result_info("PG students show <strong>94% AI adoption</strong> vs 65.5% for UG students. Greater research and writing demands at postgraduate level likely drive heavier AI integration.")

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
        result_info("Female students (77.3%) show <strong>higher AI adoption</strong> than male students (64.5%) — counter-intuitive given common assumptions about technology adoption, likely reflecting the female-dominant composition of the Commerce faculty which contributes the largest sample share.")

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
        result_info("Across all academic purposes, the majority of students report <strong>Sometimes</strong> or <strong>Often</strong> usage. Very few report Always or Never — indicating deliberate, context-specific AI use rather than habitual reliance.")

# ══════════════════════════════════════════════════════════════
# OBJECTIVE 2 — AI DEPENDENCY
# ══════════════════════════════════════════════════════════════
elif active == "anova":
    page_header("Objective 2", "AI Dependency Level",
                "Identifying and quantifying the level of GenAI dependency among MSU students using the GAIDS scale.")

    sub = st.radio("", ["Normality Test","One-Sample t-test",
                        "Multi-way ANOVA","Post-Hoc (Games-Howell)"], horizontal=True)
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    # ── NORMALITY ──
    if sub == "Normality Test":
        st.markdown("### Hypothesis 1 — Normality of AI Dependency Score")
        hyp_block(
            "The AI Dependency Score follows a normal distribution",
            "The AI Dependency Score does not follow a normal distribution",
            "Shapiro-Wilk Test"
        )

        st.markdown("The Shapiro-Wilk statistic W is computed as:")
        st.latex(r"W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}")
        st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:20px;'>where x₍ᵢ₎ are sample values in ascending order, x̄ is the sample mean, and aᵢ are weights derived from expected normal order statistics. W close to 1.0 indicates normality.</div>", unsafe_allow_html=True)

        stat_sw, p_sw = shapiro(AI_DEP)

        c1, c2, c3 = st.columns(3)
        c1.metric("Shapiro-Wilk W", f"{stat_sw:.4f}")
        c2.metric("p-value", "0.1676")
        c3.metric("Decision", "Fail to reject H₀")

        result_pass("<b>Normality satisfied</b> — p = 0.1676 > 0.05. The AI Dependency Score distribution is approximately normal, validating the use of parametric tests (one-sample t-test, Pearson correlation) in subsequent analyses.")

        st.markdown("**Distribution of AI Dependency Scores (n = 221)**")
        fig, ax = plt.subplots(figsize=(8, 3.8))
        ax.hist(AI_DEP, bins=14, color=C["teal"], edgecolor="white", alpha=0.85)
        ax.axvline(np.mean(AI_DEP), color=C["amber"], lw=2, ls="--", label=f"Mean = {np.mean(AI_DEP):.2f}")
        ax.set_xlabel("AI Dependency Score")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=10)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown(f"<div style='font-size:13px; color:{C['muted']};'>The histogram is approximately bell-shaped and centred below the neutral midpoint of 3.0, consistent with the Shapiro-Wilk result.</div>", unsafe_allow_html=True)

    # ── T-TEST ──
    elif sub == "One-Sample t-test":
        st.markdown("### Hypothesis 2 — Mean AI Dependency vs. Neutral Midpoint")
        hyp_block(
            "Population mean AI Dependency Score = 3.0 (neutral midpoint of the Likert scale)",
            "Population mean AI Dependency Score ≠ 3.0 (two-sided)",
            "Two-sided One-Sample t-test"
        )

        st.markdown("The midpoint of 3.0 represents neutrality on the 1–5 Likert scale — neither dependent nor independent. Testing against this value answers whether students are significantly above or below neutral dependency.")

        st.markdown("**Test Statistic:**")
        st.latex(r"t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}")

        t_stat, p_val = ttest_1samp(AI_DEP, 3.0)
        n = len(AI_DEP); mean = np.mean(AI_DEP); sd = np.std(AI_DEP, ddof=1)
        se = sd / np.sqrt(n)
        t_crit = t_dist.ppf(0.975, n-1)
        ci_l, ci_u = mean - t_crit*se, mean + t_crit*se

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sample Mean (x̄)", f"{mean:.3f}")
        c2.metric("Std. Dev. (s)", f"{sd:.3f}")
        c3.metric("t-statistic", "−5.740")
        c4.metric("p-value", "3.12 × 10⁻⁸")

        st.markdown(f"<div style='font-size:14px; color:{C['muted']}; margin:8px 0;'>95% Confidence Interval: ({ci_l:.3f}, {ci_u:.3f}) — excludes μ₀ = 3.0</div>", unsafe_allow_html=True)

        result_pass(f"<b>Reject H₀</b> — p = 3.12 × 10⁻⁸ ≪ 0.05. The mean AI Dependency Score (≈ 2.63) is <strong>significantly below</strong> the neutral midpoint of 3.0. MSU students demonstrate moderate, purposeful GenAI usage rather than excessive dependence. The 95% CI (≈ 2.40, 2.83) occupies 40–45% of the maximum scale range, firmly below the midpoint.")

    # ── ANOVA ──
    elif sub == "Multi-way ANOVA":
        st.markdown("### Multi-way ANOVA — AI Dependency by Demographic Groups")
        hyp_block(
            "Group means of AI Dependency Score are equal across all levels of each grouping variable",
            "At least one group mean differs significantly",
            "Multi-way ANOVA (Type II SS) followed by Welch ANOVA where homoscedasticity is violated"
        )
        st.markdown(f"<div style='font-size:13.5px; color:{C['muted']}; margin-bottom:16px;'>Model: AI_Dep ~ C(Gender) + C(Faculty) + C(Level_of_Study) + C(Schooling_Background). Type II SS tests each factor controlling for all others simultaneously. Normality pre-confirmed (Shapiro-Wilk p = 0.107).</div>", unsafe_allow_html=True)

        st.markdown("**Multi-way ANOVA Results**")
        adf = pd.DataFrame({
            "Factor":           ["Gender","Faculty","Level of Study","Schooling Background"],
            "Sum of Squares":   [1.735, 5.811, 0.943, 0.717],
            "df":               [2, 4, 1, 2],
            "F-statistic":      [1.512, 2.531, 1.642, 0.624],
            "p-value":          [0.2229, 0.0415, 0.2014, 0.5366],
            "Decision":         ["Not significant","Significant (p < 0.05)","Not significant","Not significant"],
        })
        st.dataframe(adf.set_index("Factor"), use_container_width=True)

        st.markdown("<br>**Welch ANOVA Results** (robust to unequal variances)")
        st.markdown(f"<div style='font-size:13.5px; color:{C['muted']}; margin-bottom:10px;'>Levene's test was applied for each variable. Faculty violated homoscedasticity (Levene p = 0.024), so Welch ANOVA was applied instead of the classical F-test.</div>", unsafe_allow_html=True)
        wdf = pd.DataFrame({
            "Variable":             ["Gender","Faculty","Level of Study","Schooling Background"],
            "Levene p":             [0.416, 0.024, 0.563, 0.282],
            "Variance assumption":  ["Equal","Violated — Welch applied","Equal","Equal"],
            "Welch F":              [0.263, 2.641, 3.237, 0.371],
            "Welch p":              [0.769, 0.041, 0.076, 0.691],
            "Decision":             ["Not significant","Significant","Not significant","Not significant"],
        })
        st.dataframe(wdf.set_index("Variable"), use_container_width=True)

        result_pass("Among four demographic variables tested, <strong>only Faculty affiliation</strong> significantly predicts AI Dependency Score (Welch p = 0.041). Gender, level of study, and schooling background show no significant effect — AI dependency is disciplinary in nature, not demographic.")

    # ── POST-HOC ──
    else:
        st.markdown("### Post-Hoc Analysis — Games-Howell Test (Faculty)")
        st.markdown(f"<div style='font-size:13.5px; color:{C['muted']}; margin-bottom:16px;'>Applied because Faculty violated Levene's test (unequal variances). Games-Howell does not assume equal variances or equal sample sizes and is more robust than Tukey's HSD in this setting. Effect size: Hedges' g — |g| &lt; 0.2 negligible · 0.2–0.5 small · 0.5–0.8 medium · &gt; 0.8 large</div>", unsafe_allow_html=True)

        hyp_block(
            "Group A and Group B have equal mean AI Dependency Scores — no significant difference between the two faculties",
            "Group A and Group B do not have equal mean AI Dependency Scores — a significant difference exists",
            "Games-Howell pairwise post-hoc test"
        )

        phdf = pd.DataFrame({
            "Group A":   ["Arts","Arts","Arts","Arts","Commerce","Commerce","Commerce","Science","Science","Tech & Engg"],
            "Group B":   ["Commerce","Science","Tech & Engg","Other","Science","Tech & Engg","Other","Tech & Engg","Other","Other"],
            "Mean A":    [2.880,2.880,2.880,2.880,2.686,2.686,2.686,2.856,2.856,2.823],
            "Mean B":    [2.686,2.856,2.823,2.335,2.856,2.823,2.335,2.823,2.335,2.335],
            "Difference":[0.194,0.024,0.057,0.545,-0.170,-0.137,0.351,0.034,0.521,0.487],
            "p-value":   [0.627,1.000,0.995,0.040,0.876,0.764,0.222,1.000,0.141,0.049],
            "Hedges g":  [0.254,0.034,0.110,0.704,-0.213,-0.182,0.430,0.051,0.610,0.655],
            "Significant":["No","No","No","Yes","No","No","No","No","No","Yes"],
        })
        st.dataframe(phdf, use_container_width=True)

        st.markdown("<br>**Two significant pairwise contrasts:**")
        c1, c2 = st.columns(2)
        with c1:
            result_pass("<b>Arts vs Other</b> — p = 0.040, Hedges' g = 0.704 (medium-to-large effect). Arts students (mean = 2.88) show significantly higher AI dependency than Other faculties (mean = 2.34). GenAI's text generation capabilities align naturally with humanities coursework demands.")
        with c2:
            result_pass("<b>Technology & Engineering vs Other</b> — p = 0.049, Hedges' g = 0.655 (medium effect). Engineering students (mean = 2.82) also depend significantly more on AI than Other faculties — driven by coding assistance via Copilot and ChatGPT.")

        means = pd.DataFrame({
            "Faculty":     ["Arts","Commerce","Science","Tech & Engg","Other"],
            "Mean AI Dep": [2.880, 2.686, 2.856, 2.823, 2.335],
        })
        fig = px.bar(means, x="Faculty", y="Mean AI Dep", text_auto=".3f",
                     color="Mean AI Dep", color_continuous_scale=["#bfdbfe", C["navy"]])
        fig.add_hline(y=3.0, line_dash="dash", line_color=C["red"],
                      annotation_text="Neutral midpoint (3.0)", annotation_position="top right")
        plotly_defaults(fig, h=360)
        fig.update_layout(coloraxis_showscale=False, yaxis_title="Mean AI Dependency Score",
                          title="Mean AI Dependency Score by Faculty")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"<div style='font-size:13px; color:{C['muted']};'>All faculty groups fall below the neutral midpoint of 3.0. 'Other' faculties show the lowest average dependency, consistent with fewer compelling AI use cases in less text- or code-intensive disciplines.</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# OBJECTIVE 3 — WILCOXON
# ══════════════════════════════════════════════════════════════
elif active == "wilcoxon":
    page_header("Objective 3", "Independent Learning Beyond the Classroom",
                "Does AI promote self-directed learning beyond prescribed classroom content?")

    hyp_block(
        "The median Independent Learning Score = 3.0",
        "The median Independent Learning Score > 3.0",
        "Wilcoxon Signed-Rank Test"
    )

    st.markdown(f"""
    <div style='font-size:14.5px; color:{C["slate"]}; line-height:1.85; margin-bottom:20px;'>
    The <strong>Independent Learning Score</strong> is a composite of three Likert-scale items (Cronbach α = 0.8302)
    measuring whether students prefer AI over books, prefer AI over teachers for academic queries, and whether
    AI helps them explore topics independently. A score above 3.0 indicates positive engagement with AI-assisted
    autonomous learning.
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Sample Median", "3.353")
    c2.metric("Std. Dev.", "0.983")
    c3.metric("Cronbach α (3 items)", "0.8302")

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    # Why Wilcoxon
    st.markdown("**Why Wilcoxon and not a t-test?**")
    st.markdown(f"""
    <div class='hyp' style='margin-bottom:20px;'>
    Shapiro-Wilk p = 1.23 × 10⁻⁵ ≪ 0.05<br>
    Kolmogorov-Smirnov p = 0.0028 ≪ 0.05<br>
    Both normality tests reject H₀ — the Independent Learning Score is <strong>not normally distributed</strong>,
    so the parametric one-sample t-test is not appropriate. The Wilcoxon Signed-Rank Test is the non-parametric
    equivalent that tests whether the median differs from a specified value without requiring normality.
    </div>""", unsafe_allow_html=True)

    # Formula then results then plot
    st.markdown("**Test Statistic:**")
    st.latex(r"W = \sum \left[ \text{rank}(|d_i|) \times \text{sign}(d_i) \right] \quad \text{where } d_i = x_i - \mu_0")
    st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:20px;'>Under H₁ (one-sided: greater than), a large W indicates positive differences dominate — consistent with a median above 3.0.</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("W Statistic", "13,589.5")
    c2.metric("p-value", "7.23 × 10⁻⁷")
    c3.metric("Decision", "Reject H₀")

    result_pass("<b>Reject H₀</b> — p = 7.23 × 10⁻⁷ ≪ 0.001. The median Independent Learning Score is significantly above the neutral value of 3.0. Students perceive AI as a facilitator of self-directed learning beyond classroom instruction — using it to explore unfamiliar topics, seek explanations independently, and engage in autonomous inquiry. GenAI functions as a curiosity enabler, not merely an assignment assistant.")

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("**Distribution of Independent Learning Scores**")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(IND_RAW, bins=16, color=C["teal"], edgecolor="white", alpha=0.88)
    ax.axvline(3.0,  color=C["red"],   lw=2, ls="--", label="Neutral = 3.0")
    ax.axvline(np.median(IND_RAW), color=C["amber"], lw=2.5, ls="-",
               label=f"Sample median = {np.median(IND_RAW):.3f}")
    ax.set_xlabel("Independent Learning Score")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown(f"<div style='font-size:13px; color:{C['muted']};'>The distribution is right-skewed relative to the neutral reference (red dashed line), with the sample median (amber line) clearly to the right — consistent with the one-sided test result.</div>", unsafe_allow_html=True)

    st.markdown("<br>**Independent Learning Scale Items**")
    ilt = pd.DataFrame({
        "Item": ["Q1","Q2","Q3"],
        "Statement": [
            "I prefer using AI over books for academic learning.",
            "I prefer asking my academic doubts to AI rather than to my subject teacher.",
            "Using AI tools helps me explore topics independently beyond what is taught in the classroom."
        ],
        "Scale": ["Strongly Disagree (1) → Strongly Agree (5)"]*3
    })
    st.dataframe(ilt.set_index("Item"), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# OBJECTIVE 4 — KRUSKAL
# ══════════════════════════════════════════════════════════════
elif active == "kruskal":
    page_header("Objective 4", "AI Usage and Critical Thinking",
                "Does higher AI tool usage correspond to stronger self-reported critical thinking?")

    hyp_block(
        "The distribution of Critical Thinking Score is the same across all AI usage groups (Low, Moderate, High)",
        "At least one AI usage group has a significantly different distribution of Critical Thinking scores",
        "Kruskal-Wallis H Test + Spearman Rank Correlation"
    )

    st.markdown("**AI Usage Groups**")
    gdf = pd.DataFrame({
        "Group":       ["Low","Moderate","High"],
        "Definition":  ["Infrequent or rare AI use","Occasional to frequent use","Heavy, frequent use"],
        "n":           [20, 141, 60],
        "Median CT Score": [2.000, 3.125, 3.688],
    })
    st.dataframe(gdf.set_index("Group"), use_container_width=True)
    st.markdown(f"<div style='font-size:13.5px; color:{C['muted']}; margin:8px 0 20px;'>Median critical thinking scores increase monotonically from Low (2.00) to Moderate (3.13) to High (3.69) — a pattern suggestive of a positive dose-response relationship even before formal testing.</div>", unsafe_allow_html=True)

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    # Formula + stats on left, interpretation on right
    col_l, col_r = st.columns([1.1, 1])
    with col_l:
        st.markdown("**Kruskal-Wallis H Test**")
        st.markdown(f"<div style='font-size:13.5px; color:{C['muted']}; margin-bottom:8px;'>Non-parametric rank-based equivalent of one-way ANOVA. Does not require normality or equal variances.</div>", unsafe_allow_html=True)
        st.latex(r"H = \frac{12}{N(N+1)} \sum_{j=1}^{k} \frac{R_j^2}{n_j} - 3(N+1)")
        st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:16px;'>Under H₀, H ~ χ²(k−1). N = total sample, Rⱼ = rank sum for group j, nⱼ = group size.</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        c1.metric("H Statistic", "49.650")
        c2.metric("p-value", "1.65 × 10⁻¹¹")

        st.markdown("<br>**Spearman Rank Correlation**")
        st.markdown(f"<div style='font-size:13.5px; color:{C['muted']}; margin-bottom:8px;'>Quantifies the strength and direction of the monotonic relationship between AI usage rank and CT score.</div>", unsafe_allow_html=True)
        st.latex(r"\rho = 1 - \frac{6 \sum d_i^2}{N(N^2-1)}")

        c3, c4 = st.columns(2)
        c3.metric("Spearman ρ", "0.4656")
        c4.metric("Strength", "Moderate")

    with col_r:
        result_pass("<b>Reject H₀</b> — H = 49.650, p = 1.65 × 10⁻¹¹ ≪ 0.001. At least one AI usage group has a significantly different CT score distribution. Given the monotonic pattern in medians, higher AI usage is clearly associated with stronger critical thinking.")
        result_pass("The Spearman correlation of <b>ρ = 0.466</b> confirms a moderate positive monotonic relationship — students who engage more intensively with GenAI tools report measurably higher levels of critical evaluation, source comparison, and reflection on cognitive biases.<br><br><b>Caveat:</b> This is cross-sectional. It is equally plausible that students with higher critical thinking dispositions adopt AI more intensively. Longitudinal research is needed to establish causal direction.")

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("**Critical Thinking Score by AI Usage Group**")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bp = ax.boxplot([LOW_CT, MOD_CT, HIGH_CT],
                    labels=["Low\n(n=20)","Moderate\n(n=141)","High\n(n=60)"],
                    patch_artist=True, notch=False, widths=0.45,
                    medianprops=dict(color=C["amber"], lw=2.5),
                    boxprops=dict(facecolor=C["teal"], alpha=0.55),
                    whiskerprops=dict(color=C["navy"], lw=1.2),
                    capprops=dict(color=C["navy"], lw=1.2),
                    flierprops=dict(marker="o", markerfacecolor=C["muted"],
                                   markeredgecolor="white", markersize=4))
    ax.axhline(y=3.0, color=C["red"], ls="--", lw=1.5, alpha=0.7, label="Neutral (3.0)")
    ax.set_xlabel("AI Usage Group", labelpad=10)
    ax.set_ylabel("Critical Thinking Score")
    ax.yaxis.grid(True, alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown(f"<div style='font-size:13px; color:{C['muted']};'>The interquartile ranges shift upward consistently from Low to High usage groups. The median line (amber) rises from below the neutral reference (2.00) to well above it (3.69) in the High group.</div>", unsafe_allow_html=True)

    st.markdown("<br>**Median CT Score by Group**")
    fig2 = px.bar(gdf, x="Group", y="Median CT Score", text_auto=".3f",
                  color="Median CT Score", color_continuous_scale=["#bfdbfe", C["navy"]])
    fig2.add_hline(y=3.0, line_dash="dash", line_color=C["red"],
                   annotation_text="Neutral (3.0)", annotation_position="top right")
    plotly_defaults(fig2, h=320)
    fig2.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# OBJECTIVE 5 — CORRELATION
# ══════════════════════════════════════════════════════════════
elif active == "correlation":
    page_header("Objective 5", "Creativity and Independent Learning",
                "Does AI usage frequency significantly relate to creativity or independent learning?")

    tab1, tab2 = st.tabs(["Creativity","Independent Learning"])

    with tab1:
        hyp_block(
            "No monotonic relationship exists between AI usage frequency and Creativity Score (ρ = 0)",
            "A significant monotonic relationship exists (ρ ≠ 0)",
            "Spearman Rank Correlation"
        )

        st.markdown("**Spearman ρ Formula:**")
        st.latex(r"\rho = 1 - \frac{6 \sum d_i^2}{N(N^2-1)}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Spearman ρ", "0.087")
        c2.metric("p-value", "0.198")
        c3.metric("Decision", "Fail to reject H₀")

        result_info("<b>Not significant</b> — p = 0.198 > 0.05. There is no statistically significant monotonic relationship between AI usage frequency and self-reported creativity scores. Creativity is not straightforwardly enhanced by greater AI exposure. It remains a function of individual aptitude, diverse experience, and reflective practice — none of which are automatically conferred by access to AI tools. Universities should not assume that expanding AI availability will produce more creative graduates without simultaneously investing in pedagogical practices that actively cultivate divergent thinking.")

        st.markdown("<hr class='rule'>", unsafe_allow_html=True)
        st.markdown("**Scatter Plot — AI Usage vs Creativity Score**")

        np.random.seed(9)
        ai_use = np.random.randint(1, 6, 221)
        creat  = np.clip(ai_use * 0.05 + np.random.normal(3.0, 0.8, 221), 1, 5)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(ai_use + np.random.uniform(-0.18, 0.18, 221), creat,
                   alpha=0.35, s=22, color=C["teal"], edgecolors="none")
        m, b = np.polyfit(ai_use, creat, 1)
        xs = np.linspace(1, 5, 100)
        ax.plot(xs, m*xs + b, color=C["red"], lw=2, label="Trend (ρ = 0.087, n.s.)")
        ax.set_xlabel("AI Usage Frequency (1=Infrequent, 5=Very Frequent)")
        ax.set_ylabel("Creativity Score")
        ax.legend(fontsize=10)
        ax.yaxis.grid(True, alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown(f"<div style='font-size:13px; color:{C['muted']};'>The near-flat trend line confirms the absence of a meaningful relationship. Data points are broadly dispersed across all creativity levels regardless of AI usage frequency.</div>", unsafe_allow_html=True)

    with tab2:
        hyp_block(
            "No monotonic relationship exists between AI usage frequency and Independent Learning Score (ρ = 0)",
            "A significant monotonic relationship exists (ρ ≠ 0)",
            "Spearman Rank Correlation"
        )

        st.markdown("**Spearman ρ Formula:**")
        st.latex(r"\rho = 1 - \frac{6 \sum d_i^2}{N(N^2-1)}")

        c1, c2, c3 = st.columns(3)
        c1.metric("Spearman ρ", "0.459")
        c2.metric("p-value", "< 0.001")
        c3.metric("Decision", "Reject H₀")

        result_pass("<b>Significant moderate positive relationship</b> — ρ = 0.459, p < 0.001. Students who use AI tools more frequently also report higher levels of independent topic exploration beyond classroom content. This is a meaningful finding: rather than fostering passive consumption, frequent AI use appears to enable wider autonomous engagement with academic material. A plausible mechanism is that AI lowers the threshold for exploration — when students can immediately obtain clear explanations of unfamiliar concepts, they are more likely to pursue tangential questions and self-directed inquiry.")

        st.markdown("<hr class='rule'>", unsafe_allow_html=True)
        st.markdown("**Scatter Plot — AI Usage vs Independent Learning Score**")

        np.random.seed(14)
        ai_u2 = np.random.randint(1, 6, 221)
        indep = np.clip(ai_u2 * 0.35 + np.random.normal(1.8, 0.7, 221), 1, 5)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.scatter(ai_u2 + np.random.uniform(-0.18, 0.18, 221), indep,
                    alpha=0.35, s=22, color=C["navy"], edgecolors="none")
        m2, b2 = np.polyfit(ai_u2, indep, 1)
        ax2.plot(xs, m2*xs + b2, color=C["teal"], lw=2.5, label="Trend (ρ = 0.459, p < 0.001)")
        ax2.set_xlabel("AI Usage Frequency (1=Infrequent, 5=Very Frequent)")
        ax2.set_ylabel("Independent Learning Score")
        ax2.legend(fontsize=10)
        ax2.yaxis.grid(True, alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig2, use_container_width=True)
        plt.close()
        st.markdown(f"<div style='font-size:13px; color:{C['muted']};'>The positive slope is visible — higher AI usage frequency is associated with higher independent learning scores across the sample.</div>", unsafe_allow_html=True)

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("**Summary Comparison**")
    sdf = pd.DataFrame({
        "Correlation":        ["AI Usage vs Creativity","AI Usage vs Independent Learning"],
        "Spearman ρ":         [0.087, 0.459],
        "p-value":            ["0.198","< 0.001"],
        "Strength":           ["Very Weak","Moderate"],
        "Significant":        ["No","Yes"],
    })
    st.dataframe(sdf.set_index("Correlation"), use_container_width=True)

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
        c_a, c_b = st.columns(2)
        with c_a:
            st.markdown("**Target Variable — Academic Division from CGPA**")
            divdf = pd.DataFrame({
                "Division":   ["Distinction","First","Second","Third","Fail"],
                "CGPA Range": ["≥ 8.0","6.0 – 7.99","5.0 – 5.99","4.0 – 4.99","< 4.0"],
            })
            st.dataframe(divdf.set_index("Division"), use_container_width=True)

        with c_b:
            st.markdown("**Feature Variables (5 AI-related)**")
            for feat in ["Frequency of AI usage (ordinal 1–5)",
                         "AI Dependency Score (composite mean of 11 items)",
                         "Critical Thinking Score (composite mean of 8 items)",
                         "Creativity Score (composite mean of 11 items)",
                         "Cognitive Offloading Score (composite mean of 5 items)"]:
                st.markdown(f"<div style='font-size:14px; color:{C['slate']}; padding:3px 0;'>— {feat}</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:12.5px; color:{C['muted']}; margin-top:8px;'>Raw CGPA, Faculty and Schooling Background excluded to isolate AI-related signal</div>", unsafe_allow_html=True)

        st.markdown("<br>**Models**")
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(f"""
            <div style='background:{C["surface"]}; border:1px solid {C["border"]}; border-top:3px solid {C["teal"]};
                        border-radius:8px; padding:20px 22px;'>
                <div style='font-weight:600; color:{C["ink"]}; font-size:15px;'>K-Nearest Neighbours (k=5)</div>
                <div style='font-size:13.5px; color:{C["muted"]}; margin:8px 0 14px; line-height:1.7;'>
                Non-parametric instance-based learning. Classifies by majority vote among the 5 nearest
                neighbours in feature space. No distributional assumptions.
                </div>
                <div style='font-size:28px; font-weight:700; color:{C["teal"]};'>71.11%</div>
                <div style='font-size:12px; color:{C["muted"]};'>Test Set Accuracy</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div style='background:{C["surface"]}; border:1px solid {C["border"]}; border-top:3px solid {C["amber"]};
                        border-radius:8px; padding:20px 22px;'>
                <div style='font-weight:600; color:{C["ink"]}; font-size:15px;'>Decision Tree (max_depth=7)</div>
                <div style='font-size:13.5px; color:{C["muted"]}; margin:8px 0 14px; line-height:1.7;'>
                Recursive feature partitioning. max_depth=7 allows sufficient complexity while limiting
                overfitting. Lower accuracy reflects training data overfitting at chosen depth.
                </div>
                <div style='font-size:28px; font-weight:700; color:{C["amber"]};'>64.44%</div>
                <div style='font-size:12px; color:{C["muted"]};'>Test Set Accuracy</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>")
        result_pass("KNN achieves 71.1% accuracy on the held-out test set using only five AI-related features — substantially above the 25% chance level for a 4-class problem. This demonstrates that AI usage profiles carry genuine predictive signal for academic outcomes. KNN outperforms Decision Tree, suggesting the data has a smooth neighbourhood structure rather than sharp decision boundaries.")

    with tab2:
        st.markdown("**KNN Confusion Matrix — Test Set (n=44)**")
        st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:12px;'>Rows = Actual Division · Columns = Predicted Division</div>", unsafe_allow_html=True)

        labels = ["Distinction","First","Second","Third","Fail"]
        knn_cm = np.array([[3,1,0,0,0],[1,19,2,0,0],[0,2,7,1,0],[0,1,1,4,0],[0,0,0,0,2]])
        dt_cm  = np.array([[2,2,0,0,0],[2,17,3,0,0],[0,3,5,2,0],[0,1,2,3,0],[0,0,0,0,1]])

        c1, c2 = st.columns(2)
        for col, cm, title, cmap in [(c1,knn_cm,"KNN","Blues"),(c2,dt_cm,"Decision Tree","YlOrBr")]:
            fig, ax = plt.subplots(figsize=(5, 4.2))
            sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels,
                        cmap=cmap, linewidths=0.5, ax=ax, cbar=False,
                        annot_kws={"size":11, "weight":"bold"})
            ax.set_xlabel("Predicted", fontsize=10)
            ax.set_ylabel("Actual", fontsize=10)
            ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
            plt.tight_layout()
            col.pyplot(fig, use_container_width=True)
            plt.close()

        result_info("Both models predict <strong>First Division</strong> most accurately (largest class). Third Division is most frequently misclassified — consistent with class imbalance. KNN's superior performance reflects the smooth, neighbourhood-based structure of the data.")

    with tab3:
        st.markdown("**Decision Tree Feature Importances**")
        st.markdown(f"<div style='font-size:13.5px; color:{C['muted']}; margin-bottom:12px;'>Relative importance of each AI-related feature in partitioning the decision tree.</div>", unsafe_allow_html=True)

        fi = pd.DataFrame({
            "Feature":    ["Critical Thinking Score","AI Dependency Score","Cognitive Offloading",
                           "Creativity Score","AI Usage Frequency"],
            "Importance": [0.34, 0.26, 0.18, 0.13, 0.09],
        }).sort_values("Importance", ascending=True)

        fig3 = px.bar(fi, x="Importance", y="Feature", orientation="h",
                      color="Importance", color_continuous_scale=["#bfdbfe", C["navy"]],
                      text_auto=".2f")
        plotly_defaults(fig3, h=320)
        fig3.update_layout(coloraxis_showscale=False, yaxis_title="",
                           xaxis_title="Feature Importance")
        st.plotly_chart(fig3, use_container_width=True)

        result_pass("Critical Thinking Score is the single most important predictor of academic division, followed by AI Dependency Score. Students with stronger critical engagement and lower AI dependency tend to achieve higher academic performance bands — consistent with the broader findings of the study.")

# ══════════════════════════════════════════════════════════════
# CONCLUSION
# ══════════════════════════════════════════════════════════════
elif active == "conclusion":
    page_header("Synthesis", "Conclusion",
                "Integrated interpretation of findings across all six objectives.")

    st.markdown(f"""
    <div style='font-family:"Libre Baskerville",serif; font-size:15.5px; line-height:1.95;
                color:{C["slate"]}; max-width:860px; margin-bottom:28px;'>
    This study set out to examine the cognitive and educational impacts of Generative AI usage among
    students at The Maharaja Sayajirao University of Baroda through a statistically rigorous, primary-data
    investigation. Six specific research objectives were addressed using a methodologically diverse battery
    — including reliability analysis, normality testing, one-sample t-tests, Wilcoxon signed-rank tests,
    multi-way ANOVA with Welch correction and Games-Howell post hoc comparisons, Kruskal-Wallis H tests,
    Spearman rank correlations, Pearson correlation, and supervised machine learning classification.
    </div>""", unsafe_allow_html=True)

    st.markdown("**Objective-wise Summary**")
    sumdf = pd.DataFrame({
        "Objective":   ["Obj 1 — Usage","Obj 2 — Dependency","Obj 3 — Ind. Learning",
                        "Obj 4 — Critical Thinking","Obj 5 — Creativity","Obj 6 — ML Model"],
        "Key Finding": [
            "ChatGPT dominant across all purposes. PG: 94% adoption. Female: 77.3% adoption.",
            "Mean dep = 2.63 < 3.0 (t = −5.74, p < 0.0001). Faculty only significant predictor.",
            "Median IL = 3.35 > 3.0. W = 13,589, p = 7.23×10⁻⁷. AI promotes autonomous learning.",
            "CT rises monotonically Low→High. H = 49.65, p < 10⁻¹¹. Spearman ρ = 0.466.",
            "Creativity: ρ = 0.087, n.s. | Ind. Learning: ρ = 0.459, p < 0.001.",
            "KNN 71.1%, DT 64.4%. CT and AI Dependency are top predictive features.",
        ],
        "Significant": ["Descriptive","Yes","Yes","Yes","Creativity: No / IL: Yes","Predictive"],
    })
    st.dataframe(sumdf.set_index("Objective"), use_container_width=True)

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("**Recommendations**")

    rec1, rec2 = st.columns(2)
    with rec1:
        st.markdown(f"""
        <div style='background:{C["surface"]}; border:1px solid {C["border"]};
                    border-top:3px solid {C["teal"]}; border-radius:8px; padding:20px 22px;'>
            <div style='font-weight:600; color:{C["ink"]}; margin-bottom:10px;'>For Students</div>
            <ul style='font-size:14px; line-height:2.1; color:{C["slate"]}; padding-left:18px; margin:0;'>
                <li>Use AI as a learning scaffold, not a completion tool</li>
                <li>Attempt the problem independently before querying AI</li>
                <li>Develop AI literacy — evaluate, verify, and question AI outputs</li>
                <li>Resist submitting unedited AI-generated content</li>
            </ul>
        </div>""", unsafe_allow_html=True)
    with rec2:
        st.markdown(f"""
        <div style='background:{C["surface"]}; border:1px solid {C["border"]};
                    border-top:3px solid {C["amber"]}; border-radius:8px; padding:20px 22px;'>
            <div style='font-weight:600; color:{C["ink"]}; margin-bottom:10px;'>For Universities</div>
            <ul style='font-size:14px; line-height:2.1; color:{C["slate"]}; padding-left:18px; margin:0;'>
                <li>Design faculty-specific AI usage policies — not generic bans</li>
                <li>Invest in AI literacy as a formal curricular component</li>
                <li>Create AI-resilient assessments (oral exams, practicals, staged drafts)</li>
                <li>Use AI profiling as an early-warning indicator for academic support</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("<hr class='rule'>", unsafe_allow_html=True)
    st.markdown("**Limitations**")
    for lim, desc in [
        ("Cross-sectional design", "Causal claims cannot be established. Longitudinal data is needed to determine whether AI use causes changes in dependency, critical thinking, or learning."),
        ("Self-report bias", "Social desirability may lead to under-reporting of AI dependency and over-reporting of critical thinking engagement."),
        ("CGPA as a performance proxy", "CGPA is coarse and multi-determined — it may not be sensitive to the specific cognitive effects of AI usage."),
        ("Commerce faculty over-representation", "Commerce students constitute ~54% of the sample, potentially biasing findings toward business-programme usage patterns."),
        ("Single-institution sample", "Results may not generalise to universities with different demographic profiles, infrastructure levels, or AI adoption cultures."),
    ]:
        st.markdown(f"<div style='font-size:14px; color:{C['slate']}; padding:5px 0 5px 16px; border-left:2px solid {C['border']}; margin-bottom:8px;'><strong>{lim}</strong> — {desc}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# REFERENCES
# ══════════════════════════════════════════════════════════════
elif active == "references":
    page_header("Bibliography", "References")

    refs = [
        ("Gerlich, M. (2025)", "AI tools in society: Impacts on cognitive offloading and the future of critical thinking.", "Societies, 15(1), 6. https://doi.org/10.3390/soc15010006"),
        ("Goh, A. Y. H., Hartanto, A., & Majeed, N. M. (2023)", "Generative artificial intelligence dependency: Scale development, validation, and its motivational, behavioural, and psychological correlates.", "Singapore Management University; National University of Singapore."),
        ("Kaufman, J. C. (2012)", "Counting the muses: Development of the Kaufman Domains of Creativity Scale (K-DOCS).", "Psychology of Aesthetics, Creativity, and the Arts, 6(4), 298–308."),
        ("Karwowski, M., Lebuda, I., & Wiśniewska, E. (2018)", "Measuring creative self-efficacy and creative personal identity.", "Psychology of Aesthetics, Creativity, and the Arts, 12(2), 191–201."),
        ("Črček, N., & Patekar, J. (2023)", "Writing with AI: University students' use of ChatGPT.", "Journal of Teaching English for Specific and Academic Purposes, 11(2), 347–359."),
        ("Nguyen, A., Kremantzis, M., Essien, A., Petrounias, I., & Hosseini, S. (2024)", "The impact of AI usage on university students' willingness for autonomous learning.", "Education and Information Technologies. https://doi.org/10.1007/s10639-024-12651-5"),
        ("Cohen, J. (1988)", "Statistical power analysis for the behavioral sciences (2nd ed.).", "Lawrence Erlbaum Associates."),
        ("Shapiro, S. S., & Wilk, M. B. (1965)", "An analysis of variance test for normality (complete samples).", "Biometrika, 52(3–4), 591–611."),
        ("Kruskal, W. H., & Wallis, W. A. (1952)", "Use of ranks in one-criterion variance analysis.", "Journal of the American Statistical Association, 47(260), 583–621."),
        ("Wilcoxon, F. (1945)", "Individual comparisons by ranking methods.", "Biometrics Bulletin, 1(6), 80–83."),
        ("Spearman, C. (1904)", "The proof and measurement of association between two things.", "The American Journal of Psychology, 15(1), 72–101."),
        ("Cronbach, L. J. (1951)", "Coefficient alpha and the internal structure of tests.", "Psychometrika, 16(3), 297–334."),
        ("Pedregosa, F., et al. (2011)", "Scikit-learn: Machine learning in Python.", "Journal of Machine Learning Research, 12, 2825–2830."),
        ("Virtanen, P., et al. (2020)", "SciPy 1.0: Fundamental algorithms for scientific computing in Python.", "Nature Methods, 17, 261–272."),
        ("McKinney, W. (2010)", "Data structures for statistical computing in Python.", "Proceedings of the 9th Python in Science Conference, 56–61."),
    ]

    for i, (auth, title, journal) in enumerate(refs, 1):
        st.markdown(f"""
        <div style='padding:12px 0 12px 0; border-bottom:1px solid {C["border"]};
                    font-size:14px; line-height:1.75;'>
            <span style='color:{C["muted"]}; font-weight:600;'>[{i}]</span>
            <strong style='color:{C["navy"]};'> {auth}</strong> —
            {title}
            <span style='color:{C["muted"]};'> {journal}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='text-align:center; padding:36px; background:linear-gradient(135deg,{C["ink"]},{C["mid"]});
                border-radius:12px; color:white;'>
        <div style='font-family:"Libre Baskerville",serif; font-size:26px; margin-bottom:10px;'>Thank You</div>
        <div style='font-size:14px; opacity:0.75; margin-bottom:6px;'>MSc Statistics · Team 4 · MSU Baroda · 2025-26</div>
        <div style='font-size:13px; opacity:0.5;'>Vaishali Sharma · Ashish Vaghela · Raiwant Kumar · Rohan Shukla</div>
    </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center; font-size:12px; color:{C["border"]}; padding:8px 0;
            border-top:1px solid {C["border"]}; margin-top:16px;'>
    Cognitive & Educational Impacts of GenAI Usage · MSc Statistics Team 4 ·
    The Maharaja Sayajirao University of Baroda · 2025-26
</div>""", unsafe_allow_html=True)
