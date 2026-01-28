import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import plotly.express as px
import seaborn as sns

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
# DATA VISUALIZATION
# =========================================================
elif active_tab == "📊 Data Visualization":

    st.header("Data Visualization")

    # ---------- COMMON PIE FIG ----------
    def pie_figure():
        fig, ax = plt.subplots(figsize=(3.2, 3), facecolor="#0E1117")
        ax.set_facecolor("#0E1117")
        return fig, ax

    viz_type = st.selectbox(
        "Select Visualization Type",
        [
            "Multiple Bar Chart (AI Tools vs Purpose)",
            "Pie Charts (AI Usage Distribution)",
            "AI Tools Used for Academic Purposes"
        ]
    )

    # =====================================================
    # 1. MULTIPLE BAR CHART (AI TOOLS vs PURPOSE)
    # =====================================================
    if viz_type == "Multiple Bar Chart (AI Tools vs Purpose)":

        st.subheader("Usage of AI Tools Across Academic Purposes")

        df = pd.read_excel(
            "FINAL DATA OF PROJECT (1).xlsx",
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
    # 2. PIE CHARTS (GENERAL USAGE)
    # =====================================================
    elif viz_type == "Pie Charts (AI Usage Distribution)":

        st.subheader("AI Usage Distribution (Pie Charts)")

        pie_type = st.selectbox(
            "Select Pie Chart",
            [
                "Overall AI Usage",
                "Programme-wise AI Usage",
                "Gender-wise AI Usage"
            ]
        )

        if pie_type == "Overall AI Usage":
            sizes = [159, 62]
            labels = ["Yes", "No"]
            fig, ax = pie_figure()
            ax.pie(
                sizes, labels=labels, autopct="%1.1f%%",
                startangle=90, radius=0.68,
                colors=["#4C72B0", "#DD8452"],
                wedgeprops={"edgecolor": "#1f2937"},
                textprops={"color": "white", "fontsize": 7}
            )
            ax.set_title("Overall AI Usage Among Students", color="white", fontsize=9)
            ax.axis("equal")
            st.pyplot(fig, use_container_width=False)

        elif pie_type == "Programme-wise AI Usage":
            sizes = [47, 112, 3, 59]
            labels = ["PG – Yes", "UG – Yes", "PG – No", "UG – No"]
            fig, ax = pie_figure()
            ax.pie(
                sizes, labels=labels, autopct="%1.1f%%",
                startangle=90, radius=0.68,
                colors=["#4C72B0", "#55A868", "#C44E52", "#8172B3"],
                wedgeprops={"edgecolor": "#1f2937"},
                textprops={"color": "white", "fontsize": 7}
            )
            ax.set_title("Programme-wise AI Usage Distribution", color="white", fontsize=9)
            ax.axis("equal")
            st.pyplot(fig, use_container_width=False)

        else:
            sizes = [99, 29, 60, 33]
            labels = ["Female – Yes", "Female – No", "Male – Yes", "Male – No"]
            fig, ax = pie_figure()
            ax.pie(
                sizes, labels=labels, autopct="%1.1f%%",
                startangle=90, radius=0.68,
                colors=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
                wedgeprops={"edgecolor": "#1f2937"},
                textprops={"color": "white", "fontsize": 7}
            )
            ax.set_title("Gender-wise AI Usage Distribution", color="white", fontsize=9)
            ax.axis("equal")
            st.pyplot(fig, use_container_width=False)

    # =====================================================
    # 3. ACADEMIC PURPOSE – PIE + BAR CHARTS
    # =====================================================
    elif viz_type == "AI Tools Used for Academic Purposes":

        st.subheader("AI Tools Used for Academic Purposes")

        academic_viz = st.selectbox(
            "Select Academic Visualization",
            [
                "Pie Chart – AI Tools Used",
                "Bar Chart – Most Frequently Used AI Tools",
                "Grouped Bar Chart – Frequency vs Academic Purpose"
            ]
        )

        # ---------------------------------
        # 3.1 PIE CHART (ADDED BACK ✅)
        # ---------------------------------
        if academic_viz == "Pie Chart – AI Tools Used":

            df2 = pd.read_excel(
                r"Cognitive and Educational impacts of GenAi usage among university students  (Responses).xlsx",
                sheet_name="Sheet2"
            )

            df2.columns = df2.columns.astype(str).str.strip()
            col = df2.columns[0]

            df2[col] = df2[col].replace({
                "Perplexity": "Perplexity / Copilot",
                "Copilot": "Perplexity / Copilot"
            })

            counts = df2[col].value_counts()
            labels = counts.index
            sizes = counts.values

            fig, ax = pie_figure()
            ax.pie(
                sizes,
                labels=labels,
                autopct="%1.1f%%",
                startangle=140,
                radius=0.68,
                colors=sns.color_palette("Spectral", len(labels)),
                wedgeprops={"edgecolor": "#1f2937", "linewidth": 1},
                textprops={"color": "white", "fontsize": 7},
                labeldistance=1.05,
                pctdistance=0.6
            )

            ax.set_title(
                "Distribution of AI Tools Used for Academic Purposes",
                color="white",
                fontsize=9
            )

            ax.axis("equal")
            st.pyplot(fig, use_container_width=False)

        # ---------------------------------
        # 3.2 BAR CHART – AI TOOLS FREQUENCY
        # ---------------------------------
        elif academic_viz == "Bar Chart – Most Frequently Used AI Tools":

            df2 = pd.read_excel(
                r"Cognitive and Educational impacts of GenAi usage among university students  (Responses).xlsx",
                sheet_name="Sheet2"
            )

            df2.columns = df2.columns.astype(str).str.strip()
            col = df2.columns[0]

            df2[col] = df2[col].replace({
                "Perplexity": "Perplexity / Copilot",
                "Copilot": "Perplexity / Copilot"
            })

            counts = df2[col].value_counts().reset_index()
            counts.columns = ["AI Tool", "Number of Students"]

            sns.set_theme(style="whitegrid", font_scale=1.1)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(
                data=counts,
                x="AI Tool",
                y="Number of Students",
                palette="Spectral",
                ax=ax
            )

            ax.set_title("Most Frequently Used AI Tools for Academic Purposes", fontsize=14)
            ax.set_xlabel("AI Tool")
            ax.set_ylabel("Number of Students")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")

            st.pyplot(fig)

        # ---------------------------------
        # 3.3 GROUPED BAR – FREQUENCY vs PURPOSE
        # ---------------------------------
        else:

            df3 = pd.read_excel(
                r"Cognitive and Educational impacts of GenAi usage among university students  (Responses).xlsx",
                sheet_name="Sheet3"
            )

            df3 = df3.rename(columns={df3.columns[0]: "Frequency"})

            df3_long = df3.melt(
                id_vars="Frequency",
                var_name="Usage Purpose",
                value_name="Number of Students"
            )

            sns.set_theme(style="whitegrid", font_scale=1.1)

            fig, ax = plt.subplots(figsize=(12, 7))
            sns.barplot(
                data=df3_long,
                x="Usage Purpose",
                y="Number of Students",
                hue="Frequency",
                palette="Set2",
                ax=ax
            )

            ax.set_title(
                "Frequency of GenAI Usage Across Academic Purposes",
                fontsize=15,
                weight="bold"
            )
            ax.set_xlabel("Academic Purpose")
            ax.set_ylabel("Number of Students")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
            ax.legend(title="Usage Frequency")

            st.pyplot(fig)

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
# HYPOTHESES / TESTS TAB
# =========================================================
elif active_tab == "📑 Tests":

    st.header("Statistical Hypothesis Testing")

    # =====================================================
    # HYPOTHESIS DROPDOWN
    # =====================================================
    hypothesis_list = [
        "Normality of AI Dependency Score",
        "CGPA vs AI Dependency",
        "Faculty vs AI Dependency (One-Way ANOVA)"
    ]

    selected_hypothesis = st.selectbox(
        "Select Hypothesis",
        hypothesis_list
    )

    st.markdown("---")

    # =====================================================
    # HYPOTHESIS 1: NORMALITY TEST
    # =====================================================
    if selected_hypothesis == "Normality of AI Dependency Score":

        st.subheader("Hypothesis 1: Normality of AI Dependency Score")

        st.markdown("""
        **H₀:** AI Dependency Score follows a normal distribution  
        **H₁:** AI Dependency Score does not follow a normal distribution  

        **Statistical Test Used:** Shapiro–Wilk Test  
        **Data Source:** Sheet – `Sheet5`
        """)

        df = pd.read_excel(
            "FINAL DATA OF PROJECT (1).xlsx",
            sheet_name="Sheet5"
        )
        df.columns = df.columns.astype(str).str.strip()
        dep_col = next(c for c in df.columns if "dep" in c.lower())
        ai_dep = df[dep_col].dropna()

        col1, col2 = st.columns([1.1, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            ax.hist(ai_dep, bins=8, edgecolor="black", alpha=0.75)
            ax.set_title("AI Dependency Score Distribution", fontsize=10)
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        stat, p_value = shapiro(ai_dep)

        with col2:
            st.metric("Shapiro–Wilk p-value", f"{p_value:.4f}")
            if p_value > 0.05:
                st.success("Fail to reject H₀ → Normality assumption satisfied.")
            else:
                st.warning("Reject H₀ → Data deviates from normality.")

        st.markdown("### Test Formula")
        st.latex(r"""
        W = \frac{(\sum a_i x_{(i)})^2}{\sum (x_i - \bar{x})^2}
        """)

    # =====================================================
    # HYPOTHESIS 2: CGPA vs AI DEPENDENCY
    # =====================================================
    elif selected_hypothesis == "CGPA vs AI Dependency":

        st.subheader("Hypothesis 2: CGPA vs AI Dependency")

        st.markdown("""
        **H₀:** No significant relationship exists between CGPA and AI Dependency Score  
        **H₁:** A significant relationship exists between CGPA and AI Dependency Score  

        **Statistical Test Used:** Spearman Rank Correlation  
        **Data Source:** Sheet – `AI_dep vs CGPA`
        """)

        df = pd.read_excel(
            "FINAL DATA OF PROJECT (1).xlsx",
            sheet_name="AI_dep vs CGPA"
        )
        df.columns = df.columns.astype(str).str.strip()

        cgpa = df["CGPA of Previous Semester"]
        ai_dep = df["AI_DEP_SCORE"]

        from scipy.stats import spearmanr
        rho, p_value = spearmanr(cgpa, ai_dep, nan_policy="omit")

        col1, col2 = st.columns(2)
        col1.metric("Spearman Correlation (ρ)", f"{rho:.3f}")
        col2.metric("p-value", f"{p_value:.4f}")

        strength = (
            "Negligible" if abs(rho) < 0.1 else
            "Weak" if abs(rho) < 0.3 else
            "Moderate" if abs(rho) < 0.5 else
            "Strong"
        )

        st.info(f"Observed relationship strength: **{strength} correlation**")

        st.markdown("### Test Formula")
        st.latex(r"""
        \rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}
        """)

        if p_value < 0.05:
            st.success("Reject H₀ → Significant relationship detected.")
        else:
            st.info("Fail to reject H₀ → No significant relationship detected.")

    # =====================================================
    # HYPOTHESIS 3: FACULTY vs AI DEPENDENCY (ANOVA)
    # =====================================================
    elif selected_hypothesis == "Faculty vs AI Dependency (One-Way ANOVA)":

        st.subheader("Hypothesis 3: Faculty-wise Differences in AI Dependency")

        st.markdown("""
        **H₀:** Mean AI Dependency Score is the same across all faculties  
        **H₁:** At least one faculty differs in mean AI Dependency Score  

        **Statistical Test Used:** One-Way ANOVA  
        **Data Source:** Sheet – `Form responses 1`
        """)

        df = pd.read_excel(
            "FINAL DATA OF PROJECT (1).xlsx",
            sheet_name="ANOVA"
        )
        df.columns = df.columns.astype(str).str.strip()

        if not {"Faculty", "AI_DEP_SCORE"}.issubset(df.columns):
            st.error("Required columns (Faculty, AI_DEP_SCORE) not found.")
            st.stop()

        # Group data correctly
        groups = [
            g["AI_DEP_SCORE"].dropna()
            for _, g in df.groupby("Faculty")
            if g["AI_DEP_SCORE"].dropna().shape[0] >= 2
        ]

        if len(groups) < 2:
            st.error("Not enough faculties with sufficient observations for ANOVA.")
            st.stop()

        from scipy.stats import f_oneway
        f_stat, p_value = f_oneway(*groups)

        col1, col2 = st.columns(2)
        col1.metric("F-statistic", f"{f_stat:.3f}")
        col2.metric("p-value", f"{p_value:.4f}")

        st.markdown("### Test Formula")
        st.latex(r"""
        F = \frac{\text{Between-group Mean Square}}{\text{Within-group Mean Square}}
        """)

        st.markdown("### Interpretation")

        if p_value < 0.05:
            st.success(
                "Since p-value < 0.05, we reject the null hypothesis. "
                "Mean AI Dependency Scores differ significantly across faculties."
            )
        else:
            st.info(
                "Since p-value > 0.05, we fail to reject the null hypothesis. "
                "No statistically significant difference in mean AI Dependency Scores "
                "is observed across faculties."
            )


   
# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("MSc Statistics | Dashboard Ready for Review")
