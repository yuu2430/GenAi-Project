import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro
import plotly.express as px
import seaborn as sns
from docx import Document
import os


# =====================================================
# GLOBAL COLOR THEME ( emphasize academic blue palette)
# =====================================================
THEME_COLORS = {
    "primary": "#5b9bd5",     # light academic blue
    "secondary": "#9dc3e6",   # softer blue
    "accent1": "#bdd7ee",     # very light blue
    "accent2": "#ddebf7",     # pastel background blue
    "accent3": "#2f75b5",     # slightly darker (for contrast)
    "neutral": "#f4f8fc"      # near-white
}

# Bars: light → slightly darker (projector readable)
BAR_COLORS = [
    "#5b9bd5",
    "#9dc3e6",
    "#bdd7ee",
    "#2f75b5"
]

# Pies: soft pastel sequence
PIE_COLORS = [
    "#5b9bd5",
    "#9dc3e6",
    "#bdd7ee",
    "#ddebf7"
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
    "📝 Questionnaire",
    "📊 Data Visualization",
    "📑 Inference"
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

    1. To find out how often students use AI tools and for what purpose, and see how this differs by their education level, or gender.  
    <br>

    2. To understand how students’ views on AI dependence differ across different age groups, genders, and study programs.  
    <br>

    3. To check if depending on AI for thinking (cognitive offloading) affects the link between AI usage and students’ learning performance (CGPA).  
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
    st.info("“Has Generative AI impacted your education?”")

    st.subheader("Pilot Results Summary")
    st.markdown("""
    - **Yes:** 48 students  
    - **No:** 10 students  
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
# QUESTIONNAIRE
# =========================================================

elif active_tab == "📝 Questionnaire":

    st.header("Research Questionnaire")
    st.caption("Survey Instrument Used for Data Collection")
    st.markdown("---")

    from docx import Document

    @st.cache_data
    def load_questionnaire():
        doc = Document("Questionnaire.docx")
        content = []
        for para in doc.paragraphs:
            if para.text.strip():
                content.append(para.text)
        return content

    questionnaire_text = load_questionnaire()

    for line in questionnaire_text:
        if line.lower().startswith("section"):
            st.subheader(line)
        elif line.lower().startswith("reference"):
            st.markdown(f"📌 **{line}**")
        elif line.lower().startswith("description"):
            st.markdown(f"**{line}**")
        else:
            st.write(line)

    st.markdown("---")
    st.markdown("📄 **Full Questionnaire (Google Form PDF):**  "
                "[Click here to view on GitHub]"
                "(https://github.com/yuu2430/GenAi-Project/blob/main/"
                "Cognitive%20and%20Educational%20impacts%20of%20GenAi%20usage%20among%20university%20students%20-%20Google%20Forms.pdf)")




# =========================================================
# DATA VISUALIZATION
#=========================================================
elif active_tab == "📊 Data Visualization":

    st.header("Objective 1: To find out how often students use AI tools and for what purpose, and see how this differs by their education level, or gender.")

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

        st.subheader("Frequency of AI Used for Academic Purposes")
        st.markdown("### Select Tool Visualization")

        col_x, col_y, col_z = st.columns(3)

        with col_x:
            if st.button("🥧 Pie – AI Tools Used", use_container_width=True):
                st.session_state.academic_viz = "Pie Chart – AI Tools Used"

        

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
elif active_tab == "📑 Inference":

    st.header("Objective 2: To identify the level of dependence on GenAI among MSU students.")

    # =====================================================
    # HYPOTHESIS DROPDOWN
    # =====================================================
    hypothesis_list = [
        "Normality of AI Dependency Score",
        "Mean AI Dependency vs Neutral Value (One-Sample t-test)",
        "CGPA vs AI Dependency (Pearson Correlation)"
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
        *H₀:* AI Dependency Score follows a normal distribution  
        *H₁:* AI Dependency Score does not follow a normal distribution  

        *Statistical Test Used:* Shapiro–Wilk Test  
        *Data Source:* FINAL DATA OF PROJECT (1).xlsx → Sheet5
        """)

        df = pd.read_excel(
            "FINAL DATA OF PROJECT (1).xlsx",
            sheet_name="Sheet5"
        )
        df.columns = df.columns.astype(str).str.strip()
        dep_col = next(c for c in df.columns if "dep" in c.lower())
        scores = df[dep_col].dropna()

        col1, col2 = st.columns([1.1, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(4.5, 3))
            ax.hist(scores, bins=8, edgecolor="black", alpha=0.75)
            ax.set_title("AI Dependency Score Distribution", fontsize=10)
            ax.set_xlabel("Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)

        stat, p_value = shapiro(scores)

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
    # HYPOTHESIS 2: ONE-SAMPLE T-TEST
    # =====================================================
    elif selected_hypothesis == "Mean AI Dependency vs Neutral Value (One-Sample t-test)":

        st.subheader("Hypothesis 2: Mean AI Dependency vs Neutral Value")

        st.markdown("""
        *H₀:* Population mean AI Dependency Score = 3  
        *H₁:* Population mean AI Dependency Score ≠ 3  

        *Statistical Test Used:* Two-sided One-Sample t-test  
        *Significance Level:* α = 0.05  
        *Data Source:* FINAL DATA OF PROJECT (1).xlsx → Sheet5
        """)

        df = pd.read_excel(
            "FINAL DATA OF PROJECT (1).xlsx",
            sheet_name="Sheet5"
        )
        df.columns = df.columns.astype(str).str.strip()
        dep_col = next(c for c in df.columns if "dep" in c.lower())
        scores = df[dep_col].dropna()

        mu_0 = 3
        alpha = 0.05

        n = len(scores)
        mean = np.mean(scores)
        std = np.std(scores, ddof=1)

        from scipy.stats import ttest_1samp, t
        t_stat, p_value = ttest_1samp(scores, mu_0)

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Sample Size (n)", n)
            st.metric("Sample Mean", f"{mean:.3f}")
            st.metric("Sample Std. Dev.", f"{std:.3f}")

        with col2:
            st.metric("t-statistic", f"{t_stat:.3f}")
            st.metric("p-value (two-tailed)", f"{p_value:.10f}")

        st.markdown("### Interpretation")

        if p_value < alpha:
            direction = "greater than" if mean > mu_0 else "less than"
            st.success(
                f"Since the p-value ({p_value:.10f}) is less than the significance level "
                f"α = {alpha}, the null hypothesis is rejected. "
                f"This indicates that the mean AI dependency score is "
                f"statistically significantly {direction} the hypothesized value of {mu_0}. "
                "Thus, there is strong evidence that the population mean differs from the neutral value.")
        else:
            st.info(
                "Fail to reject H₀ → No significant difference from the neutral value."
            )


        # Confidence Interval
        t_crit = t.ppf(1 - alpha / 2, df=n - 1)
        margin = t_crit * (std / np.sqrt(n))
        ci_lower = mean - margin
        ci_upper = mean + margin

        st.markdown("### 95% Confidence Interval for Mean")
        st.info(f"({ci_lower:.3f}, {ci_upper:.3f})")
        st.markdown("### Interpretation")
        if ci_lower <= mu_0 <= ci_upper:
            st.success(
                f"The 95% confidence interval for the mean AI dependency score "
                f"includes the hypothesized value μ₀ = {mu_0}. "
                "This indicates that, at the 5% level of significance, "
                "there is insufficient evidence to conclude that the population mean "
                "AI dependency score differs significantly from the neutral value."
                "This indicates that university students’ average level of dependence on GenAI for academic purposes lies between 40.05% and 45.14% of the total scale range.")
        else:
            direction = "greater than" if mean > mu_0 else "less than"
            st.info(
                f"The 95% confidence interval for the mean AI dependency score "
                f"does not include the hypothesized value μ₀ = {mu_0}. "
                f"This suggests that the population mean AI dependency score is "
                f"significantly {direction} the neutral value at the 5% level of significance."
                "This indicates that university students’ average level of dependence on GenAI for academic purposes lies between 40.05% and 45.14% of the total scale range.")


        

        
    # =====================================================
    # HYPOTHESIS 3: PEARSON CORRELATION
    # =====================================================
    elif selected_hypothesis == "CGPA vs AI Dependency (Pearson Correlation)":

        st.subheader("Hypothesis 3: CGPA vs AI Dependency")

        st.markdown("""
        *H₀:* No significant linear relationship exists between CGPA and AI Dependency Score  
        *H₁:* A significant linear relationship exists between CGPA and AI Dependency Score  

        *Statistical Test Used:* Pearson's Correlation  
        *Data Source:* FINAL DATA OF PROJECT (1).xlsx → AI_dep vs CGPA
        """)

        df = pd.read_excel(
            "FINAL DATA OF PROJECT (1).xlsx",
            sheet_name="AI_dep vs CGPA"
        )
        df.columns = df.columns.astype(str).str.strip()

        cgpa = df["CGPA of Previous Semester"]
        ai_dep = df["AI_DEP_SCORE"]

        from scipy.stats import pearsonr
        r, p_value = pearsonr(cgpa.dropna(), ai_dep.dropna())

        col1, col2 = st.columns(2)
        col1.metric("Pearson Correlation (r)", f"{r:.3f}")
        col2.metric("p-value", f"{p_value:.4f}")

        strength = (
            "Negligible" if abs(r) < 0.1 else
            "Weak" if abs(r) < 0.3 else
            "Moderate" if abs(r) < 0.5 else
            "Strong"
        )

        direction = "positive" if r > 0 else "negative"

        st.info(
            f"Observed relationship: *{strength} {direction} linear correlation*"
        )

        st.markdown("### Test Formula")
        st.latex(r"""
        r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}
                 {\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}
        """)

        if p_value < 0.05:
            st.success("Reject H₀ → Significant linear relationship detected.")
        else:
            st.info("Fail to reject H₀ → No significant linear relationship detected.")
# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("MSc Statistics | Dashboard Ready for Review")
