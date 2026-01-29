import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import plotly.express as px
import seaborn as sns

# =====================================================
# GLOBAL COLOR THEME ( emphasize academic blue palette)
# =====================================================
THEME_COLORS = {
    "primary": "#04408d",
    "secondary": "#055cb8",
    "accent1": "#5b97e1",
    "accent2": "#a0bce4",
    "accent3": "#267ed7",
    "neutral": "#e6eef8"
}

BAR_COLORS = [
    THEME_COLORS["primary"],
    THEME_COLORS["secondary"],
    THEME_COLORS["accent3"],
    THEME_COLORS["accent1"]
]

PIE_COLORS = [
    THEME_COLORS["primary"],
    THEME_COLORS["accent1"],
    THEME_COLORS["accent2"],
    THEME_COLORS["accent3"]
]


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
    "📑 Tests"
]

active_tab = st.sidebar.radio(
    "Navigation",
    tabs
)



# =========================================================
# OVERVIEW
# =========================================================
if active_tab == "📘 Overview":
    st.title("Cognitive and Learning Implications of Generative AI Usage Among University Students")
    st.markdown("""
    **MSc Statistics – 5 Credit Project**  
    **Institution:** The Maharaja Sayajirao University of Baroda  
    **Students:**  Rohan Shukla • Vaishali Sharma • Raiwant Kumar • Ashish Vaghela  
    **Mentor:** Murlidharan Kunnumal""")
    st.caption("Project Dashboard")
    st.markdown("---")

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

    st.markdown("""
    <div style="font-size:18px; line-height:1.8;">

    1. To find out how often students use AI tools and for what purpose, and see how this differs by their education level, subject area, or gender.  
    <br>

    2. To understand how students’ views on AI dependence differ across different age groups, genders, and study programs.  
    <br>

    3. To check if depending on AI for thinking (cognitive offloading) affects the link between AI usage and students’ learning performance or depth of understanding.  
    <br>

    4. To study how AI tool usage is related to students’ critical thinking and engagement in learning, using self-reported ratings.  
    <br>

    5. To explore how using AI tools influences students’ academic skills, like creativity, and independent learning.  
    <br>

    6. To create a model that predicts what factors lead to positive or negative effects of AI tool usage on students’ academic outcomes.  

    </div>
    """, unsafe_allow_html=True)

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
#=========================================================
elif active_tab == "📊 Data Visualization":

    st.header("Data Visualization")

    # =====================================================
    # SESSION STATE
    # =====================================================
    if "viz_type" not in st.session_state:
        st.session_state.viz_type = "Multiple Bar Chart – AI Tools vs Academic Purpose"

    if "academic_viz" not in st.session_state:
        st.session_state.academic_viz = "Pie Chart – AI Tools Used"

    # =====================================================
    # MAIN NAVIGATION
    # =====================================================
    st.markdown("### Select Visualization Type")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("📊 AI Tools vs Academic Purpose", use_container_width=True):
            st.session_state.viz_type = "Multiple Bar Chart – AI Tools vs Academic Purpose"

    with col_b:
        if st.button("🥧 Programme & Gender Distribution", use_container_width=True):
            st.session_state.viz_type = "AI Usage Distribution (Programme & Gender)"

    with col_c:
        if st.button("🛠️ AI Tools Used", use_container_width=True):
            st.session_state.viz_type = "AI Tools Used for Academic Purposes"

    viz_type = st.session_state.viz_type
    st.markdown("---")

    # =====================================================
    # 1. MULTIPLE BAR CHART
    # =====================================================
    if viz_type == "Multiple Bar Chart – AI Tools vs Academic Purpose":

        st.subheader("Usage of AI Tools Across Academic Purposes")

        df = pd.read_excel("FINAL DATA OF PROJECT (1).xlsx", sheet_name="Sheet3")
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
                        "Academic Purpose": purpose,
                        "AI Tool": tool,
                        "Number of Students": int(subset[tool].sum())
                    })

        final_df = pd.DataFrame(records)

        fig = px.bar(
            final_df,
            x="Academic Purpose",
            y="Number of Students",
            color="AI Tool",
            barmode="group",
            text_auto=True,
            template="plotly_white",
            color_discrete_sequence=BAR_COLORS
        )

        fig.update_layout(
            height=550,
            font=dict(color=THEME_COLORS["primary"])
        )

        st.plotly_chart(fig, use_container_width=True)

    # =====================================================
    # 2. PROGRAMME & GENDER DISTRIBUTION
    # =====================================================
    elif viz_type == "AI Usage Distribution (Programme & Gender)":

        st.subheader("Programme-wise AI Usage")

        col1, col2 = st.columns(2)
        # UG

        with col1:
            ug_df = pd.DataFrame({"Usage": ["Yes", "No"], "Count": [112, 59]})
            fig_ug = px.pie(
                ug_df,
                names="Usage",
                values="Count",
                title="UG Students",
                hole=0.45,
                template="plotly_white",
                color_discrete_sequence=[
                    THEME_COLORS["primary"],
                    THEME_COLORS["accent2"]
                ]
            )
            st.plotly_chart(fig_ug, use_container_width=True)

        with col2:
            pg_df = pd.DataFrame({"Usage": ["Yes", "No"], "Count": [47, 3]})
            fig_pg = px.pie(
                pg_df,
                names="Usage",
                values="Count",
                title="PG Students",
                hole=0.45,
                template="plotly_white",
                color_discrete_sequence=[
                    THEME_COLORS["primary"],
                    THEME_COLORS["accent2"]
                ]
            )
            st.plotly_chart(fig_pg, use_container_width=True)

        st.markdown("---")
        st.subheader("Gender-wise AI Usage")

        col3, col4 = st.columns(2)

        with col3:
            female_df = pd.DataFrame({"Usage": ["Yes", "No"], "Count": [99, 29]})
            fig_female = px.pie(
                female_df,
                names="Usage",
                values="Count",
                title="Female Students",
                hole=0.45,
                template="plotly_white",
                color_discrete_sequence=[
                    THEME_COLORS["secondary"],
                    THEME_COLORS["accent3"]
                ]
            )
            st.plotly_chart(fig_female, use_container_width=True)

        with col4:
            male_df = pd.DataFrame({"Usage": ["Yes", "No"], "Count": [60, 33]})
            fig_male = px.pie(
                male_df,
                names="Usage",
                values="Count",
                title="Male Students",
                hole=0.45,
                template="plotly_white",
                color_discrete_sequence=[
                    THEME_COLORS["secondary"],
                    THEME_COLORS["accent3"]
                ]
            )
            st.plotly_chart(fig_male, use_container_width=True)

    # =====================================================
    # 3. AI TOOLS USED FOR ACADEMIC PURPOSES
    # =====================================================
    else:

        st.subheader("AI Tools Used for Academic Purposes")
        st.markdown("### Select Tool Visualization")

        col_x, col_y, col_z = st.columns(3)

        with col_x:
            if st.button("🥧 Pie – AI Tools Used", use_container_width=True):
                st.session_state.academic_viz = "Pie Chart – AI Tools Used"

        with col_y:
            if st.button("📊 Bar – Most Used Tools", use_container_width=True):
                st.session_state.academic_viz = "Bar Chart – Most Frequently Used AI Tools"

        with col_z:
            if st.button("📈 Grouped Bar – Purpose vs Frequency", use_container_width=True):
                st.session_state.academic_viz = "Grouped Bar Chart – Frequency vs Academic Purpose"

        academic_viz = st.session_state.academic_viz

        # ---------------- PIE ----------------
        if academic_viz == "Pie Chart – AI Tools Used":

            df2 = pd.read_excel(
                "Cognitive and Educational impacts of GenAi usage among university students  (Responses).xlsx",
                sheet_name="Sheet2"
            )

            col = df2.columns[0]
            df2[col] = df2[col].replace(
                {"Perplexity": "Perplexity / Copilot", "Copilot": "Perplexity / Copilot"}
            )

            counts = df2[col].value_counts().reset_index()
            counts.columns = ["AI Tool", "Number of Students"]

            fig = px.pie(
                counts,
                names="AI Tool",
                values="Number of Students",
                hole=0.45,
                template="plotly_white",
                color_discrete_sequence=PIE_COLORS
            )

            st.plotly_chart(fig, use_container_width=True)

        # ---------------- BAR ----------------
        elif academic_viz == "Bar Chart – Most Frequently Used AI Tools":

            df2 = pd.read_excel(
                "Cognitive and Educational impacts of GenAi usage among university students  (Responses).xlsx",
                sheet_name="Sheet2"
            )

            col = df2.columns[0]
            df2[col] = df2[col].replace(
                {"Perplexity": "Perplexity / Copilot", "Copilot": "Perplexity / Copilot"}
            )

            counts = df2[col].value_counts().reset_index()
            counts.columns = ["AI Tool", "Number of Students"]

            fig = px.bar(
                counts,
                x="AI Tool",
                y="Number of Students",
                text_auto=True,
                template="plotly_white",
                color_discrete_sequence=[THEME_COLORS["primary"]]
            )

            st.plotly_chart(fig, use_container_width=True)

        # ---------------- GROUPED BAR ----------------
        else:

            df3 = pd.read_excel(
                "Cognitive and Educational impacts of GenAi usage among university students  (Responses).xlsx",
                sheet_name="Sheet3"
            )

            df3 = df3.rename(columns={df3.columns[0]: "Frequency"})

            df_long = df3.melt(
                id_vars="Frequency",
                var_name="Academic Purpose",
                value_name="Number of Students"
            )

            fig = px.bar(
                df_long,
                x="Academic Purpose",
                y="Number of Students",
                color="Frequency",
                barmode="group",
                text_auto=True,
                template="plotly_white",
                color_discrete_sequence=BAR_COLORS
            )

            fig.update_layout(height=550)
            st.plotly_chart(fig, use_container_width=True)

            
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
