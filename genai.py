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
st.sidebar.title("📊 Team 4 Project")
st.sidebar.caption("MSc Statistics – Mid-Sem Project Presentation")

tabs = [
    "📘 Overview",
    "🎯 Objectives",
    "🧪 Pilot Survey",
    "📐 Sampling & Sample Size",
    "📝 Questionnaire",
    "📋 Dataset Overview",
    "📊 Data Visualization",
    "📑 Inference",
    "📌 Conclusion",
    "📚 References"
    
]

active_tab = st.sidebar.radio(
    "",
    tabs
)



# =========================================================
# OVERVIEW
# =========================================================
if active_tab == "📘 Overview":

    # ---------- HERO TITLE ----------
    st.markdown(
        f"""
        <div style="
            margin-top: 30px;
            background: linear-gradient(90deg,
                {THEME_COLORS['primary']},
                {THEME_COLORS['secondary']});
            padding: 26px 30px;
            border-radius: 14px;
            color: black;
        ">
            <h2 style="margin-bottom:8px;">
            Cognitive & Educational Impacts of Generative AI Usage<br>
            Among University Students
            </h2>
            <p style="font-size:16px; opacity:0.95;">
            MSc Statistics · Mid-Sem Project Presentation
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------- PROJECT INFO CARD ----------
    st.markdown(
        f"""
        <div style="
            background-color:{THEME_COLORS['accent2']};
            padding:18px 22px;
            border-radius:12px;
            font-size:16px;
            line-height:1.7;
        ">
        <b>Programme:</b> MSc Statistics (Team 4)<br>
        <b>Institution:</b> The Maharaja Sayajirao University of Baroda<br>
        <b>Students:</b> Rohan Shukla • Vaishali Sharma • Raiwant Kumar • Ashish Vaghela<br>
        <b>Mentor:</b> Prof. Murlidharan Kunnumal
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ---------- OVERVIEW CONTENT ----------
    st.header("Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            f"""
            <div style="
                background-color:{THEME_COLORS['neutral']};
                padding:18px;
                border-radius:12px;
                border-left:5px solid {THEME_COLORS['primary']};
            ">
            <h4>Introduction</h4>
            <p style="font-size:16px; line-height:1.7;">
            Generative Artificial Intelligence (GenAI) tools such as ChatGPT, Gemini, and Copilot
            have become integral to university learning environments. These tools assist students
            in content creation, coding, idea generation, and exam preparation.<br><br>

            As GenAI becomes embedded in academic workflows, it is important to understand its
            influence on cognitive engagement, learning depth, motivation, and independent thinking.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div style="
                background-color:{THEME_COLORS['neutral']};
                padding:18px;
                border-radius:12px;
                border-left:5px solid {THEME_COLORS['accent3']};
            ">
            <h4>Why This Project?</h4>
            <p style="font-size:16px; line-height:1.7;">
            Existing research on GenAI largely focuses on usability and performance.
            The deeper <b>cognitive and educational impacts</b> remain under-explored.<br><br>

            This study addresses this gap through a structured <b>statistical analysis</b>
            of AI usage patterns, learning behaviour, and academic outcomes.
            </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    # ---------- AIM ----------
    st.subheader("Aim of the Study")
    st.markdown(
        f"""
        <div style="
            background-color:{THEME_COLORS['accent2']};
            padding:18px 22px;
            border-radius:12px;
            font-size:17px;
            line-height:1.7;
            border-left:5px solid {THEME_COLORS['primary']};
        ">
        To examine the cognitive and educational impacts of Generative AI usage among
        university students, with emphasis on learning behaviour, academic performance,
        and critical thinking abilities.
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# OBJECTIVES
# =========================================================
elif active_tab == "🎯 Objectives":

    st.header("Objectives of the Study")
    st.markdown("---")

    objectives = [
        "To find out how often students use AI tools and for what purpose, and see how this differs by their education level or gender.",
        "To identify the level of dependence on GenAI among MSU students.",
        "To check if depending on AI for thinking (cognitive offloading) affects the link between AI usage and students’ learning performance (CGPA).",
        "To study how AI tool usage is related to students’ critical thinking and engagement in learning, using self-reported ratings.",
        "To explore how using AI tools influences students’ academic skills, such as creativity and independent learning.",
        "To create a model that predicts factors leading to positive or negative effects of AI tool usage on students’ academic outcomes."
    ]

    for i, obj in enumerate(objectives, start=1):
        st.markdown(
            f"""
            <div style="
                background-color: {THEME_COLORS['accent2']};
                padding: 14px 18px;
                border-radius: 10px;
                margin-bottom: 12px;
                font-size: 17px;
                line-height: 1.7;
                border-left: 5px solid {THEME_COLORS['primary']};
            ">
            <b>Objective {i}.</b> {obj}
            </div>
            """,
            unsafe_allow_html=True
        )

# =========================================================
# SAMPLING & SAMPLE SIZE
# =========================================================
elif active_tab == "📐 Sampling & Sample Size":

    st.header("Sampling Design and Sample Size Determination")
    st.markdown("---")


    col1, col2 = st.columns(2)

    with col1:
        st.metric("Sampling Method", "Probability Proportional to Size (PPS)")
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
        st.latex(r"n = \frac{z_{\alpha/2}^2 \, p \, q}{E^2}")
        st.markdown("**Where:**")
        st.latex(r"z_{\alpha/2} = 1.96")
        st.markdown("Critical value for 95% confidence level")
        st.latex(r"p = 0.827")
        st.markdown("Assumed population proportion")
        st.latex(r"q = 1 - p")
        st.markdown("Margin of error")
        st.latex(r"E = 0.05")
        
        
    st.subheader("Proportional Allocation Principle")

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
        st.metric("**Total Population (N)**", "37,095")
        st.metric("**Sampling Method**", "Simple Random Sampling")

    with col2:
        st.metric("**Pilot Sample Size (n)**", "58")
        st.metric("**Pilot Question**", "Single Perception Item")

    st.markdown("---")

    st.subheader("Pilot Survey Question")
    st.info("“**Has Generative AI impacted your education?**”")

    st.subheader("Pilot Results Summary")
    
    
    col3, col4 = st.columns(2)
    with col3:
        st.metric("**Yes**", "48")

    with col4:
        st.metric("**No**", "10")

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

    st.header("Questionnaire Overview")
    st.markdown("---")

    from docx import Document
    import os

    def load_questionnaire_fresh(file_path):
        """
        Loads questionnaire content fresh on every rerun.
        No caching to avoid stale section issues.
        """
        doc = Document(file_path)
        content = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                content.append(text)
        return content

    # 🔹 File name must EXACTLY match GitHub
    QUESTIONNAIRE_FILE = "Questionnaire.docx"

    if not os.path.exists(QUESTIONNAIRE_FILE):
        st.error("Questionnaire file not found in GitHub repository.")
    else:
        questionnaire_text = load_questionnaire_fresh(QUESTIONNAIRE_FILE)

        for line in questionnaire_text:

            # SECTION HEADINGS
            if line.lower().startswith("section"):
                st.subheader(line)

            # PART HEADINGS (Part A, Part B, etc.)
            elif line.lower().startswith("part"):
                st.markdown(f"### {line}")

            # DESCRIPTION LABEL
            elif line.lower().startswith("description"):
                st.markdown(f"**{line}**")

            # REFERENCE LABEL
            elif line.lower().startswith("reference"):
                st.markdown(f"📌 **{line}**")

            # LINKS
            elif line.startswith("http"):
                st.markdown(f"[{line}]({line})")

            # NORMAL TEXT
            else:
                st.write(line)

    st.markdown("---")

    # 🔗 GitHub hyperlink to full questionnaire PDF
    st.markdown(
        "📄 **Full Questionnaire (Google Form PDF):** "
        "[Click here]"
        "(https://github.com/yuu2430/GenAi-Project/blob/main/"
        "Cognitive%20and%20Educational%20impacts%20of%20GenAi%20usage%20among%20university%20students%20-%20Google%20Forms.pdf)"
    )

# =========================================================
# DATASET OVERVIEW
# =========================================================
# =========================================================
# DATASET OVERVIEW
# =========================================================
elif active_tab == "📋 Dataset Overview":

    st.header("Dataset Overview")
    st.markdown("---")

    FILE_NAME = "FINAL DATA OF PROJECT (1).xlsx"

    try:
        # Read Excel file
        excel_file = pd.ExcelFile(FILE_NAME)

        # Show available sheets (for transparency)
        sheet_names = excel_file.sheet_names

        # Auto-pick the first sheet (Google Forms responses)
        SHEET_NAME = sheet_names[0]

        st.info(f"Using sheet: **{SHEET_NAME}**")

        df = pd.read_excel(FILE_NAME, sheet_name=SHEET_NAME)

        # Remove Timestamp column safely
        df = df.drop(columns=["Timestamp"], errors="ignore")

        # ===============================
        # BASIC DATASET INFO
        # ===============================
        col1, col2, col3 = st.columns(3)
        col1.metric("Number of Responses", df.shape[0])
        col2.metric("Number of Variables", df.shape[1])
        col3.metric("Total Missing Values", int(df.isnull().sum().sum()))

        st.markdown("---")

        # ===============================
        # DATA PREVIEW
        # ===============================
        st.subheader("Form Responses Preview")
        st.dataframe(df, use_container_width=True)

        
    except Exception as e:
        st.error("Unable to load Excel file.")
        st.code(str(e))

# =========================================================
# DATA VISUALIZATION
#=========================================================
elif active_tab == "📊 Data Visualization":

    st.header("Objective 1: To find out how often students use AI tools and for what purpose, and see how this differs by their education level, or gender.")
    st.markdown("---")


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
    #st.markdown("### Select Visualization Type")

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
        #st.markdown("### Select Tool Visualization")

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
    # SESSION STATE FOR BUTTON SELECTION
    # =====================================================
    if "selected_hypothesis" not in st.session_state:
        st.session_state.selected_hypothesis = "Normality of AI Dependency Score"

    # =====================================================
    # HYPOTHESIS BUTTONS (REPLACED DROPDOWN)
    # =====================================================
    st.subheader("")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Normality of AI Dependency Score",use_container_width=True):
            st.session_state.selected_hypothesis = "Normality of AI Dependency Score"

    with col2:
        if st.button("Mean AI Dependency vs Neutral Value (One-Sample t-test)",use_container_width=True):
            st.session_state.selected_hypothesis = "Mean AI Dependency vs Neutral Value (One-Sample t-test)"

    with col3:
        if st.button("CGPA vs AI Dependency (Pearson Correlation)",use_container_width=True):
            st.session_state.selected_hypothesis = "CGPA vs AI Dependency (Pearson Correlation)"

    selected_hypothesis = st.session_state.selected_hypothesis

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
# CONCLUSION
# =========================================================
# =========================================================
# CONCLUSION
# =========================================================
elif active_tab == "📌 Conclusion":

    st.header("Conclusion")
    #st.caption("Integrated Summary of Findings")
    st.markdown("---")

    conclusion_df = pd.DataFrame({
        "Analysis Section": [
            "Pilot Survey Findings",
            "AI Tools Across Academic Purposes",
            "Programme-wise & Gender-wise Usage",
            "Frequency of AI Usage",
            "Normality of AI Dependency Score",
            "Mean AI Dependency vs Neutral Value",
            "CGPA vs AI Dependency"
        ],

        "Key Result": [
            "Majority of students perceive AI as impactful on education.",
            "ChatGPT is the most widely used AI tool across academic tasks.",
            "PG students and female students report comparatively higher AI usage.",
            "Most students use AI tools occasionally to frequently, not continuously.",
            "AI dependency scores follow an approximately normal distribution.",
            "Mean AI dependency score is significantly lower than the neutral value (3).",
            "No significant linear relationship exists between AI dependency and CGPA."
        ],

        "Statistical Evidence": [
            "Pilot proportion p = 0.827",
            "Descriptive frequency counts and grouped bar charts",
            "Programme-wise and gender-wise pie charts",
            "Grouped bar charts across frequency levels",
            "Shapiro–Wilk test: p-value > 0.05",
            "One-sample t-test: p-value < 0.05; 95% CI excludes 3",
            "Pearson correlation: r ≈ 0; p-value > 0.05"
        ],

        "Conclusion / Implication": [
            "The research problem is relevant and empirically supported.",
            "GenAI tools primarily function as academic support tools.",
            "AI adoption varies across demographic and academic contexts.",
            "Students exhibit balanced, need-based AI usage behaviour.",
            "Parametric statistical methods are appropriate for inference.",
            "Students show moderate dependence on AI, not excessive reliance.",
            "Academic performance is influenced by multiple factors beyond AI use."
        ]
    })

    st.dataframe(conclusion_df, use_container_width=True)

    st.markdown("---")

    st.subheader("Overall Conclusion")

    st.info(
        "This study concludes that Generative AI tools have become an integral part of "
        "students’ academic workflows at The Maharaja Sayajirao University of Baroda. "
        "While GenAI tools significantly enhance learning efficiency, accessibility, and "
        "academic support, students demonstrate controlled and purposeful usage rather than "
        "excessive dependence. Statistical analysis reveals that AI dependency does not "
        "significantly influence academic performance (CGPA), highlighting the continued "
        "importance of individual effort, critical thinking, and independent learning in "
        "higher education."
    )


    
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; font-size:42px; color:black; margin-top:10px;'>"
        "<b>Thank you! :)</b>"
        "</div>",
        unsafe_allow_html=True
    )

#references
elif active_tab == "📚 References":

    st.header("References")

    st.markdown(
f"""
<div style="
    background-color:{THEME_COLORS['neutral']};
    padding:22px;
    border-radius:14px;
    font-size:16px;
    line-height:1.8;
    border-left:6px solid {THEME_COLORS['primary']};
">

<ol>

<li>
Gerlich, M. (2025). <i>AI tools in society: Impacts on cognitive offloading and the future of critical thinking</i>.
<b>Societies, 15</b>(1), 6.
<a href="https://doi.org/10.3390/soc15010006" target="_blank">
https://doi.org/10.3390/soc15010006
</a>
</li><br>

<li>
Goh, A. Y. H., Hartanto, A., &amp; Majeed, N. M. (2023).
<i>Generative artificial intelligence dependency: Scale development, validation, and its motivational, behavioral, and psychological correlates</i>.
Singapore Management University; National University of Singapore.
</li><br>

<li>
Kaufman, J. C. (2012).
<i>Counting the muses: Development of the Kaufman Domains of Creativity Scale (K-DOCS)</i>.
<b>Psychology of Aesthetics, Creativity, and the Arts, 6</b>(4), 298–308.
</li><br>

<li>
Karwowski, M., Lebuda, I., &amp; Wiśniewska, E. (2018).
<i>Measuring creative self-efficacy and creative personal identity</i>.
<b>Psychology of Aesthetics, Creativity, and the Arts, 12</b>(2), 191–201.
</li><br>

<li>
Črček, N., &amp; Patekar, J. (2023).
<i>Writing with AI: University students’ use of ChatGPT</i>.
</li><br>

<li>
<i>The impact of AI usage on university students’ willingness for autonomous learning</i>.
</li>

</ol>
</div>
""",
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    st.info(
        "📌 References are formatted in APA style and aligned with the theoretical framework of this study."
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption("Team 4 - MSc Statistics - Final Year")
