import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="GenAI Impact Study – MSc Statistics",
    layout="wide"
)

# =========================================================
# GLOBAL CSS — CLEAN & ACADEMIC
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.2rem;
}

/* Navigation bar */
div[role="radiogroup"] {
    display: flex;
    justify-content: center;
    gap: 0.6rem;
    margin-top: 0.4rem;
    margin-bottom: 0.2rem;
}

div[role="radiogroup"] label {
    background: rgba(255,255,255,0.04);
    padding: 7px 14px;
    border-radius: 10px;
    font-weight: 500;
    border: 1px solid rgba(255,255,255,0.08);
}

div[role="radiogroup"] label[data-checked="true"] {
    background: rgba(120,140,255,0.18);
    border-color: rgba(120,140,255,0.45);
    color: #e6e9ff;
}

input[type="radio"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================
st.title("Cognitive and Learning Implications of Generative AI Usage Among University Students")

st.markdown("""
**MSc Statistics – 5 Credit Project**  
**Institution:** The Maharaja Sayajirao University of Baroda  

**Students:**  
Rohan Shukla • Vaishali Sharma • Raiwant Kumar • Ashish Vaghela  

**Mentor:** Murlidharan Kunnumal  
""")

st.caption("Interactive Research Dashboard • Content & Design Finalised")
st.markdown("---")

# =========================================================
# NAVIGATION
# =========================================================
st.sidebar.title("📊 GenAI Project")
st.sidebar.caption("MSc Statistics – Research Dashboard")

tabs = [
    "📘 Overview",
    "🎯 Objectives",
    "📐 Sampling & Sample Size",
    "🧪 Pilot Survey",
    "📊 Data Visualization",
    "🧠 Reliability",
    "📑 Tests",
    "🚀 What Next?"
]

active_tab = st.sidebar.radio(
    "Navigation",
    tabs
)



# =========================================================
# OVERVIEW
# =========================================================
if active_tab == "📘 Overview":

    st.header("Project Overview")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Introduction")
        st.write("""
        Generative Artificial Intelligence (GenAI) tools such as ChatGPT, Gemini, and Copilot
        have become integral to university learning environments. These tools assist students
        in content creation, coding, idea generation, and exam preparation.

        As GenAI becomes embedded in academic workflows, it is important to understand its
        influence on cognitive engagement, learning depth, motivation, and independent thinking.
        """)

    with col2:
        st.subheader("Why This Project?")
        st.write("""
        Existing research on GenAI largely focuses on usability and performance.
        The deeper **cognitive and educational impacts** remain under-explored.

        This study addresses this gap through a structured **statistical analysis**
        of AI usage patterns, learning behaviour, and academic outcomes.
        """)

    st.markdown("---")

    st.subheader("Aim of the Study")
    st.info("""
    To examine the cognitive and educational impacts of Generative AI usage among
    university students, with emphasis on learning behaviour, academic performance,
    and critical thinking abilities.
    """)

# =========================================================
# OBJECTIVES
# =========================================================
elif active_tab == "🎯 Objectives":

    st.header("Objectives of the Study")

    obj_tabs = st.tabs(["Primary Objectives", "Analytical & Applied Objectives"])

    with obj_tabs[0]:
        st.markdown("""
        1. Quantify the frequency and purpose of GenAI usage among students  
        2. Examine the relationship between AI usage and cognitive engagement  
        3. Study the mediating role of cognitive offloading in learning outcomes  
        """)

    with obj_tabs[1]:
        st.markdown("""
        4. Identify differences in AI dependence across demographic categories  
        5. Assess the impact of AI on creativity, problem-solving, and independent learning  
        6. Develop a predictive statistical model for positive vs negative academic effects  
        """)

# =========================================================
# SAMPLING & SAMPLE SIZE
# =========================================================
elif active_tab == "📐 Sampling & Sample Size":

    st.header("Sampling Design and Sample Size Determination")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Sampling Method", "Convenience Sampling")
        st.metric("Population", "MSU Students")
        st.metric("Final Sample Size", "221")

    with col2:
        st.metric("Confidence Level", "95%")
        st.metric("Significance Level (α)", "0.05")
        st.metric("Data Type", "Primary")

    st.markdown("---")

    st.subheader("Faculty-wise, Gender-wise and Programme-wise Sample Distribution")

    sampling_data = pd.DataFrame({
        "Faculty": [
            "Arts", "Commerce", "Education & Psychology", "Family & Community Sciences",
            "Fine Arts", "Journalism & Communication", "Law", "Management Studies",
            "Performing Arts", "Pharmacy", "Science", "Social Work", "Technology & Engineering"
        ],
        "Total": [25, 120, 5, 7, 4, 1, 8, 2, 2, 2, 23, 3, 19],
        "Female UG": [13, 58, 3, 5, 2, 1, 4, 1, 1, 1, 9, 1, 3],
        "Female PG": [3, 8, 1, 1, 1, 0, 0, 0, 0, 0, 4, 1, 2],
        "Male UG": [8, 50, 1, 1, 1, 0, 4, 1, 1, 1, 8, 1, 12],
        "Male PG": [1, 4, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2]
    })

    st.dataframe(sampling_data, use_container_width=True)

    with st.expander("Sample Size Formula (Single Proportion)"):
        st.latex(r"n = \frac{Z^2 \cdot p \cdot q}{e^2}")
        st.markdown("""
        Where:  
        - \( Z = 1.96 \) (95% confidence level)  
        - \( p = 0.5 \), \( q = 1 - p \)  
        - \( e = 0.05 \)  

        Using \( p = 0.5 \) ensures maximum variability and a conservative estimate.
        """)

    st.subheader("Proportional Allocation Principle (Theoretical Basis)")

    st.latex(r"n_i = \frac{N_i}{N} \times n")

    st.markdown("""
    **Where:**  
    - \( n_i \): Sample size for the *i-th* stratum  
    - \( N_i \): Population size of the *i-th* stratum  
    - \( N \): Total population (37,095)  
    - \( n \): Total sample size (221)  

    Proportional allocation was used as a **guiding framework** to ensure
    fair representation across faculties. The table above shows the
    **actual achieved sample**, classified by faculty, gender, and level of study (UG/PG).
    """)


# =========================================================
# PILOT SURVEY
# =========================================================
elif active_tab == "🧪 Pilot Survey":

    st.header("Pilot Survey Analysis")

    st.success("Pilot survey conducted prior to full-scale data collection.")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Population (N)", "37,095")
        st.metric("Pilot Sample Size (n)", "58")
        st.metric("Sampling Method", "Simple Random Sampling")

    with col2:
        st.metric("Positive Impact", "48")
        st.metric("Negative / No Impact", "10")
        st.metric("Pilot Question", "Single Perception Item")

    st.markdown("---")

    st.subheader("Pilot Survey Question")
    st.info("“How has Generative AI impacted your education?”")

    st.subheader("Pilot Results Summary")
    st.markdown("""
    - **Positive impact:** 48 students  
    - **Negative or no impact:** 10 students  
    """)

    st.subheader("Estimated Proportion from Pilot Study")
    st.latex(r"p = \frac{48}{58} = 0.827")
    st.latex(r"q = 1 - p = 0.173")

    with st.expander("Importance of Pilot Study"):
        st.markdown("""
        - Tested clarity and relevance of questionnaire  
        - Improved construct grouping  
        - Validated Likert-scale consistency  
        - Provided empirical estimate of population proportion  
        - Reduced measurement and response bias  
        """)

# =========================================================
# DATA
# DATA VISUALIZATION
# =========================================================
# =========================================================
# DATA VISUALIZATION
# =========================================================
elif active_tab == "📊 Data Visualization":

    st.header("Data Visualization")

    import plotly.express as px
    import matplotlib.pyplot as plt

    # -----------------------------------------------------
    # MAIN DROPDOWN: TYPE OF VISUALIZATION
    # -----------------------------------------------------
    viz_type = st.selectbox(
        "Select Visualization Type",
        [
            "Multiple Bar Chart (AI Tools vs Purpose)",
            "Pie Charts (AI Usage Distribution)"
        ]
    )

    # =====================================================
    # OPTION 1: MULTIPLE BAR CHART
    # =====================================================
    if viz_type == "Multiple Bar Chart (AI Tools vs Purpose)":

        st.subheader("Usage of AI Tools Across Academic Purposes")

        df = pd.read_excel(
            "FINAL DATA OF PROJECT.xlsx",
            sheet_name="Sheet3"
        )

        df.columns = df.columns.astype(str).str.strip()
        df = df.rename(columns={df.columns[0]: "Label"})

        purposes = [
            "Project / Assignment",
            "Concept Learning",
            "Writing / Summarizing",
            "Exam Preparation",
            "Research / Idea Generation",
            "Programming / Coding"
        ]

        ai_tools = ["ChatGPT", "Gemini", "Copilot", "Perplexity"]

        df["Purpose"] = df["Label"].where(df["Label"].isin(purposes)).ffill()

        df = df[
            (~df["Label"].isin(purposes)) &
            (~df["Label"].str.contains("Grand Total", na=False))
        ]

        records = []
        for purpose in purposes:
            subset = df[df["Purpose"] == purpose]
            for tool in ai_tools:
                if tool in subset.columns:
                    records.append({
                        "Purpose": purpose,
                        "AI Tool": tool,
                        "Count": int(subset[tool].sum())
                    })

        final_df = pd.DataFrame(records)

        fig = px.bar(
            final_df,
            x="Purpose",
            y="Count",
            color="AI Tool",
            barmode="group",
            text_auto=True,
            title="Usage of AI Tools Across Academic Purposes"
        )

        fig.update_layout(
            xaxis_title="Academic Purpose",
            yaxis_title="Number of Students",
            legend_title="AI Tool",
            height=550
        )

        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # OPTION 2: PIE CHARTS
    # =====================================================
    else:

        st.subheader("AI Usage Distribution (Pie Charts)")

        pie_type = st.selectbox(
            "Select Pie Chart",
            [
                "Overall AI Usage",
                "Programme-wise AI Usage",
                "Gender-wise AI Usage"
            ]
        )

        # ---------- FIXED PIE FIG ----------
        def pie_figure():
            fig, ax = plt.subplots(
                figsize=(4.2, 3.8),   # ✅ laptop friendly
                facecolor="#0E1117"
            )
            ax.set_facecolor("#0E1117")
            return fig, ax

        # ================================
        # OVERALL AI USAGE
        # ================================
        if pie_type == "Overall AI Usage":

            sizes = [159, 62]
            labels = ["Yes", "No"]

            fig, ax = pie_figure()
            ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                radius=0.68,
                colors=["#4C72B0", "#DD8452"],
                wedgeprops={"edgecolor": "#1f2937", "linewidth": 1},
                textprops={"color": "white", "fontsize": 9}
            )

            ax.set_title("Overall AI Usage Among Students", color="white", fontsize=12)
            ax.axis("equal")

            st.pyplot(fig, use_container_width=False)

        # ================================
        # PROGRAMME-WISE AI USAGE
        # ================================
        elif pie_type == "Programme-wise AI Usage":

            sizes = [47, 112, 3, 59]
            labels = ["PG – Yes", "UG – Yes", "PG – No", "UG – No"]

            fig, ax = pie_figure()
            ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                radius=0.68,
                colors=["#4C72B0", "#55A868", "#C44E52", "#8172B3"],
                wedgeprops={"edgecolor": "#1f2937", "linewidth": 1},
                textprops={"color": "white", "fontsize": 9}
            )

            ax.set_title("Programme-wise AI Usage Distribution", color="white", fontsize=12)
            ax.axis("equal")

            st.pyplot(fig, use_container_width=False)

        # ================================
        # GENDER-WISE AI USAGE
        # ================================
        else:

            sizes = [99, 29, 60, 33]
            labels = ["Female – Yes", "Female – No", "Male – Yes", "Male – No"]

            fig, ax = pie_figure()
            ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
                radius=0.68,
                colors=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
                wedgeprops={"edgecolor": "#1f2937", "linewidth": 1},
                textprops={"color": "white", "fontsize": 9}
            )

            ax.set_title("Gender-wise AI Usage Distribution", color="white", fontsize=12)
            ax.axis("equal")

            st.pyplot(fig, use_container_width=False)

# =========================================================
# RELIABILITY
# =========================================================
elif active_tab == "🧠 Reliability":

    st.header("Reliability Analysis")

    st.info("Cronbach’s Alpha will be used to assess internal consistency.")

    with st.expander("Interpretation Guidelines"):
        st.markdown("""
        - α ≥ 0.70 → Good reliability  
        - 0.60 ≤ α < 0.70 → Acceptable  
        - α < 0.60 → Requires revision  
        """)

    with st.expander("Constructs Tested"):
        st.markdown("""
        - Cognitive Offloading  
        - Academic Behaviour  
        - Critical Thinking  
        - Creativity  
        - Withdrawal  
        """)

# =========================================================
# HYPOTHESES
# =========================================================
elif active_tab == "📑 Tests":

    st.header("Tests")

    st.markdown("""
    **H₀:** AI Dependency Score is not Normal.  

    **H₁:** AI Dependency Score is Normal.
    """)
# ---------- NORMALITY TEST ----------
    st.markdown("---")
    st.subheader("Normality Assessment of AI Dependency Score")

    dep_df = pd.read_excel(
        "FINAL DATA OF PROJECT.xlsx",
        sheet_name="Sheet5"
    )

    dep_df.columns = dep_df.columns.astype(str).str.strip()

    dep_col = next((c for c in dep_df.columns if "dep" in c.lower()), None)
    dep_score = dep_df[dep_col].dropna()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(dep_score, bins=8, edgecolor="black")
    ax.set_title("Histogram of AI Dependency Score")
    st.pyplot(fig)

    stat, p_value = shapiro(dep_score)
    st.write(f"Shapiro–Wilk p-value: {p_value:.4f}")
    st.markdown("""
    **Interpretation:** Since the p-value is greater than the significance level α = 0.05, we fail to reject the null hypothesis.
                There is no statistically significant evidence to suggest that the AI Dependency Score deviates from normality.
                 Hence, the AI Dependency Score can be considered approximately normally distributed. 
    """)

# =========================================================
# WHAT NEXT
# =========================================================
elif active_tab == "🚀 What Next?":

    st.header("Future Scope and Deliverables")

    st.markdown("""
    - Inferential statistical testing  
    - Correlation and regression analysis  
    - Group-wise comparisons (UG vs PG, faculty-wise)  
    - Final dashboard, statistical report, and presentation  
    """)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("MSc Statistics | Dashboard Ready for Review")
