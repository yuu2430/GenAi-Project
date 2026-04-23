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
st.latex(r"F = \frac{\text{MSSB}}{\text{MSSW}} = \frac{\sum_i n_i(\bar{x}_i - \bar{x})^2 / (k-1)}{\sum_i\sum_j(x_{ij}-\bar{x}_i)^2 / (N-k)}")
notation("F = F-statistic · MSSB = mean sum of square between groups · MSSW = mean sum of square within groups · x̄ᵢ = group mean · x̄ = grand mean · k = number of groups (3) · N = total sample size")
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


step("Step 3 — Assumption Decision")
assumption_decision("Both normality and equal variance violated → <strong>Kruskal-Wallis H Test</strong> (non-parametric) is appropriate.")

step("Step 4 — Final Test: Kruskal-Wallis H Test")
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
"Arts":[1.000,0.2228,0.96,0.8077,0.0027],
"Commerce":[0.2228,1.000,0.2779,0.3711,0.0126],
"Science":[0.96,0.2779,1.000,0.6646,0.0090],
"Tech & Engg":[0.8077,0.3711,0.6646,1.000,0.0030],
"Other":[0.0027,0.0126,0.0090,0.0030,1.000],
})
st.dataframe(dunn_df.set_index(""), use_container_width=True)
st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:14px;'>p-values are Bonferroni-corrected. Significant pairs: p < 0.05.</div>", unsafe_allow_html=True)

step("Significant Pairwise Differences")
        c1,c2,c3,c4 = st.columns(1)
        c1,c2,c3,c4 = st.columns(4)
with c1:
result_pass("<b>Arts vs Other</b> — p = 0.0.0027 (Significant difference)")
with c2:
@@ -1270,7 +1270,7 @@
notation("H = Kruskal-Wallis statistic · N = total observations (221) · k = number of groups (3) · nⱼ = size of group j (20, 141, 60) · Rⱼ = sum of ranks for group j · Under H₀: H ~ χ²(k−1) = χ²(2)")
c1,c2,c3 = st.columns(3)
c1.metric("H Statistic","2.6965"); c2.metric("p-value","0.2597"); c3.metric("Decision","Fail to Reject H₀")
    result_info("<b>Fail to Reject H₀</b> — p = 0.2597. No significant difference in CGPA across Low, Moderate, and High AI usage groups. AI usage level does not significantly predict academic performance.")
    result_info("<b>Fail to Reject H₀</b> — p = 0.2597. No significant difference in CGPA across Low, Moderate, and High AI usage groups.")

st.markdown("<hr class='rule'>", unsafe_allow_html=True)
st.markdown("**CGPA Distribution by AI Usage Group**")
@@ -1306,12 +1306,7 @@

st.dataframe(snippet_ct, use_container_width=True)

    # Summary stats
    gdf = pd.DataFrame({"AI Usage Group":["Low","Moderate","High"],"n":[20,141,60],"Mean CT Score":[2.025,3.063,3.735],"Median CT Score":[2.000,3.125,3.688]})
    st.markdown("**Group Summary:**")
    st.dataframe(gdf.set_index("AI Usage Group"), use_container_width=True)
    st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:6px;'>Critical Thinking Score: composite of 8 Likert items (scale 1–5). A clear increasing pattern is visible in the medians: Low < Moderate < High.</div>", unsafe_allow_html=True)

    
fig,ax = plt.subplots(figsize=(9,4.2))
ax.boxplot(
[LOW_CT, MOD_CT, HIGH_CT],
@@ -1356,7 +1351,7 @@
notation("Z = standardised JT statistic · μⱼ = expected value of J under H₀ · σⱼ = standard deviation of J under H₀ · N = total sample (221) · nᵢ = group sizes (20, 141, 60)")
c1,c2,c3,c4 = st.columns(4)
c1.metric("J Statistic","9,532.0"); c2.metric("Z Value","7.0705"); c3.metric("p-value","< 0.0001"); c4.metric("Decision","Reject H₀ ✓")
    result_pass("<b>Reject H₀</b> — Z = 7.07, p < 0.0001. Highly significant ordered trend confirmed: Low (Mdn=2.00) < Moderate (Mdn=3.13) < High (Mdn=3.69). Higher AI usage associates with higher critical thinking scores.")
    result_pass("<b>Reject H₀</b> —  p < 0.0001. Highly significant ordered trend confirmed: Low < Moderate < High . Higher AI usage associates with higher critical thinking scores.")


# ══════════════════════════════════════════════════════════════
@@ -1374,8 +1369,7 @@
step("Step 1 — Data Snippet")
snip_c = pd.DataFrame({"Respondent":["R1","R2","R3","R4","R5"],"Creativity Score":[3.636,3.455,3.545,3.636,4.636]})
st.dataframe(snip_c, use_container_width=True)
        st.markdown(f"<div style='font-size:13px; color:{C['muted']}; margin-bottom:6px;'>n = 221, Mean = 2.898, Median = 3.000, SD = 0.892. Creativity Score = mean of 11 K-DOCS items (scale 1–5). Score of 3.0 = neutral impact.</div>", unsafe_allow_html=True)

        
step("Step 2 — Normality Check")
hyp_block("The Creativity Score follows a normal distribution","The Creativity Score does NOT follow a normal distribution","Shapiro-Wilk Test")
st.latex(r"W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}")
@@ -1436,7 +1430,7 @@
assumption_decision("Normality violated → <strong>Wilcoxon Signed-Rank Test</strong> (non-parametric) is used.")

step("Step 4 — Final Test: Wilcoxon Signed-Rank Test (one-sided, greater)")
        hyp_block("Median Ind_learning Score = 3.0 (neutral)","Median Ind_learning Score > 3.0 — AI promotes independent learning","Wilcoxon Signed-Rank Test")
        hyp_block("Median Ind_learning Score = 3.0 (neutral)","Median Ind_learning Score > 3.0 — AI supports independent learning","Wilcoxon Signed-Rank Test")
st.latex(r"W = \sum \left[ \text{rank}(|d_i|) \times \text{sign}(d_i) \right], \quad d_i = x_i - 3.0")
notation("W = Wilcoxon signed-rank statistic · dᵢ = Ind_learning scoreᵢ − 3.0 · rank(|dᵢ|) = rank of absolute deviation from neutral · sign(dᵢ) = direction of deviation")
c1,c2,c3 = st.columns(3)
@@ -1447,7 +1441,7 @@
st.markdown("### Comparative Summary — Creativity vs Independent Learning")
comp_df = pd.DataFrame({"Construct":["Creativity Score","Ind_learning Score"],"Median":[3.000,3.333],"W Statistic":["9,509.0","13,589.5"],"p-value":["0.9107","7.23 × 10⁻⁷"],"Conclusion":["Fail to Reject H₀ — No enhancement","Reject H₀ — AI promotes learning ✓"]})
st.dataframe(comp_df.set_index("Construct"), use_container_width=True)
    result_info("AI does <b>not</b> enhance creativity (neutral), but <b>significantly promotes</b> independent learning. The educational impact of AI is construct-specific.")
    result_info("AI does <b>not</b> enhance creativity (neutral), but <b>significantly supports</b> independent learning.")


# ══════════════════════════════════════════════════════════════
@@ -1496,8 +1490,8 @@


result_pass("""<b>Key Findings:</b><br><br>
    • Critical Thinking is the strongest splitting feature at the root node<br>
    • AI Dependency and Creativity appear at deeper splits<br>
    • AI Dependency is the strongest splitting feature at the root node<br>
    • Critical Thinking and AI usage appear at deeper splits<br>
   • KNN (71.1%) outperforms Decision Tree (64.4%) — pattern similarity matters more than rule-based separation<br>
   • Academic performance can be reasonably predicted from AI-related cognitive features""")

@@ -1514,7 +1508,7 @@
   </div>""", unsafe_allow_html=True)
k1,k2,k3,k4,k5,k6 = st.columns(6)
k1.metric("Students Surveyed","221"); k2.metric("Faculties","13"); k3.metric("Research Objectives","6")
    k4.metric("AI Tools Benchmarked","5"); k5.metric("JAM Questions / Tool","25"); k6.metric("KNN Model Accuracy","71.1%")
    k4.metric("AI Tools Benchmarked","5"); k5.metric("JAM Questions","25"); k6.metric("KNN Model Accuracy","71.1%")
st.markdown("<hr class='rule'>", unsafe_allow_html=True)
st.markdown("### Findings by Research Objective")
obj_data = [
