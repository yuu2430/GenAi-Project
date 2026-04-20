Supplementary Section: AI Tool Accuracy and Performance Evaluation

1. Overview and Study Design
As a supplementary investigation to the primary research objectives of this project, a structured accuracy and performance evaluation of five prominent Generative AI tools was conducted across five academic subject domains. The primary purpose of this evaluation is to contextualise the AI tools that students in the main survey rely upon - and to objectively assess how well these tools actually perform on university-level academic questions, independent of student self-report.
The five AI tools evaluated are: ChatGPT (OpenAI), Microsoft Copilot, Perplexity AI, Google Gemini, and Claude (Anthropic). The five subject domains tested are: Physics, Mathematical Science, Mathematics, Chemistry, and Biotechnology. These domains were selected to span both qualitative and quantitative academic disciplines, spanning conceptual explanation, numerical problem-solving, and applied scientific reasoning.
1.1 Measurement Variables
For each combination of tool and subject, five questions were posed and four performance dimensions were recorded:

Variable	Scale / Unit	Definition
Accuracy	YES / NO / 0.5 (2 Answers)	Whether the AI tool provided a factually correct response. 'YES' = fully correct; 'NO' = incorrect or incomplete; '2 Answers' / 'GAVE 2 ANSWERS' = ambiguous dual response coded as 0.5.
Response Time	Seconds (s)	Time taken from query submission to complete response generation, measured in seconds. Lower is faster.
Detailed Response	YES / NO	Whether the response was substantively detailed - i.e., provided explanation, working, or context rather than a bare one-line answer.
Re-prompting Required	YES / NO	Whether the AI required a follow-up prompt or rephrasing of the original question before producing an acceptable answer. 'YES' indicates the first-attempt response was insufficient.

1.2 Scoring Methodology
Accuracy was the primary outcome variable. It was calculated as the percentage of the 5 questions per subject for which the AI provided a correct response (partial credit of 0.5 was assigned to responses coded as '2 Answers' - cases where the AI gave two alternative answers without identifying the correct one). Response time, detail rate, and re-prompting rate are secondary performance indicators that contextualise the accuracy findings and inform a composite assessment of each tool.
Colour Key in Tables:  Green = Strong performance   Orange = Moderate   Red = Weak performance
Accuracy thresholds: ≥ 90% = Green   |   70–89% = Orange   |   < 70% = Red   Response time: < 9s = Green   |   9–14s = Orange   |   ≥ 15s = Red
 
2. Comprehensive Performance Table - All Tools × All Subjects
The master table below presents the complete performance data for every combination of AI tool and academic subject, recording accuracy, average response time, detailed response rate, and re-prompting requirement. Each subject’s five-question battery is summarised into a single row per tool. The bottom section reports overall averages computed across all five subjects.

Subject	AI Tool	Accuracy	Avg. Response Time	Detailed Response Rate	Needed Re-prompting
Physics	ChatGPT	100%	14s	40%	No
	Copilot	100%	8.17s	40%	Yes (20%)
	Perplexity	60%	6.67s	0%	No
	Gemini	100%	10.62s	0%	Yes (20%)
	Claude	100%	8.56s	0%	No
Mathematical Science	ChatGPT	80%	19.22s	100%	No
	Copilot	100%	17.57s	80%	No
	Perplexity	40%	10.69s	0%	No
	Gemini	100%	16.81s	20%	No
	Claude	100%	12.76s	80%	No
Mathematics	ChatGPT	80%	19.53s	100%	No
	Copilot	100%	15.04s	100%	No
	Perplexity	100%	5.28s	0%	No
	Gemini	100%	13.61s	0%	No
	Claude	100%	12.77s	60%	No
Chemistry	ChatGPT	40%	13.89s	100%	No
	Copilot	60%	10.92s	80%	No
	Perplexity	60%	6.3s	0%	No
	Gemini	100%	10.96s	60%	No
	Claude	40%	12.68s	100%	No
Biotechnology	ChatGPT	100%	12.24s	100%	No
	Copilot	80%	8.44s	100%	No
	Perplexity	80%	4.43s	0%	No
	Gemini	80%	11.78s	100%	No
	Claude	80%	11.93s	60%	No
Overall Average	Accuracy	Avg. Time	Detail Rate	Re-prompt %
	ChatGPT	80%	15.78s	88%	No
	Copilot	88%	12.03s	80%	Yes (4%)
	Perplexity	68%	6.67s	0%	No
	Gemini	96%	12.76s	36%	Yes (4%)
	Claude	84%	11.74s	60%	No

n = 5 questions per tool per subject (25 questions per tool total; 125 observations total across all tools).
 
3. Accuracy Analysis
3.1 Subject-wise Accuracy Matrix
The matrix below consolidates accuracy scores by subject and tool for rapid cross-comparison. Each cell is colour-coded: green (≥ 90%), orange (70–89%), or red (< 70%).

Subject / Domain	ChatGPT	Copilot	Perplexity	Gemini	Claude
Physics	100%	100%	60%	100%	100%
Mathematical Science	80%	100%	40%	100%	100%
Mathematics	80%	100%	100%	100%	100%
Chemistry	40%	60%	60%	100%	40%
Biotechnology	100%	80%	80%	80%	80%
Overall	80%	88%	68%	96%	84%

3.2 Tool-wise Accuracy Interpretation
Gemini - Highest overall accuracy (96.0%). 
Gemini achieved perfect accuracy (100%) in four out of five subjects - Physics, Mathematical Science, Mathematics, and Chemistry. It was the only tool to score 100% in Chemistry, which proved to be the most discriminating subject for all tools. Its sole relative weakness is Biotechnology (80%), where all tools except ChatGPT also scored below perfect. Gemini's consistent correctness across both conceptual (Physics, Biology) and computational (Mathematics) domains suggests the broadest reliability profile of all tools tested.
Copilot - Second highest accuracy (88.0%). 
Copilot matched Gemini's performance in Mathematical Science and Mathematics (both 100%) and performed strongly in Physics (100%). Its accuracy drops to 60% in Chemistry and 80% in Biotechnology. Notably, Copilot was the only tool that occasionally required re-prompting (in Physics, 20% of questions needed follow-up), suggesting that while its final answers were largely correct, its first-attempt responses were sometimes insufficient in scope or specificity.
Claude - Third in accuracy (84.0%), jointly with Copilot in multiple subjects. 
Claude achieved perfect accuracy in Physics, Mathematical Science, and Mathematics - matching the performance of the top tools in purely technical and quantitative domains. However, it scores poorly in Chemistry (40%), tying with ChatGPT at the bottom for that subject. Claude's strength appears to lie in STEM-adjacent reasoning and mathematical logic, where its structured outputs are well-matched to question requirements.
ChatGPT - Fourth in accuracy (80.0%). 
ChatGPT performed perfectly in Physics and Biotechnology but showed notable weakness in Chemistry (40%) and reduced accuracy in Mathematical Science and Mathematics (both 80%). Chemistry appears to be ChatGPT's most significant accuracy gap - three of five chemistry questions received incorrect responses. This is despite ChatGPT producing the highest proportion of detailed responses (88% across all subjects), which suggests that verbosity does not compensate for factual accuracy gaps.
Perplexity - Lowest accuracy (68.0%). 
Perplexity recorded the lowest overall accuracy. It scored only 60% in Physics, 40% in Mathematical Science, and 60% in Chemistry. Its sole full accuracy score (100%) was in Mathematics, suggesting it handles direct numerical computation reasonably well but struggles with conceptual scientific reasoning and multi-step derivations. Perplexity also produced no detailed responses across any subject - all of its answers were concise and factual in style, which may partially explain lower accuracy on complex questions requiring explanatory depth.
 
4. Response Time Analysis
4.1 Subject-wise Response Time Matrix (in seconds)
The matrix below shows the average response time per tool per subject. Green cells (≤ 9s) indicate fast responses; orange (9–14s) moderate; red (≥ 15s) slow.

Subject / Domain	ChatGPT	Copilot	Perplexity	Gemini	Claude
Physics	14s	8.17s	6.67s	10.62s	8.56s
Mathematical Science	19.22s	17.57s	10.69s	16.81s	12.76s
Mathematics	19.53s	15.04s	5.28s	13.61s	12.77s
Chemistry	13.89s	10.92s	6.3s	10.96s	12.68s
Biotechnology	12.24s	8.44s	4.43s	11.78s	11.93s
Overall Avg.	15.78s	12.03s	6.67s	12.76s	11.74s

4.2 Response Time Interpretation
Perplexity - Fastest by a large margin (avg. 6.67s). 
Perplexity consistently produces responses in under 11 seconds across all subjects, with averages as low as 4.43 seconds for Biotechnology and 5.28 seconds for Mathematics. This speed advantage is substantial - it is more than twice as fast as ChatGPT on average. However, this speed comes at a direct cost: Perplexity's accuracy is the lowest of all tools. The absence of any detailed responses suggests Perplexity is optimised for rapid, concise factual retrieval rather than comprehensive academic explanation.
Claude - Second fastest overall (avg. 11.74s). 
Claude occupies a strong middle ground: competitive speed without the accuracy sacrifice seen in Perplexity. Its average response times range from 8.56 seconds (Physics) to 12.77 seconds (Mathematics), suggesting consistent performance without extreme outliers. This speed-accuracy balance positions Claude as one of the most practically efficient tools for academic use.
Copilot - Third in speed (avg. 12.03s), close to Claude. 
Copilot's response times are competitive in Physics (8.17s) and Biotechnology (8.44s) but rise substantially for Mathematical Science (17.57s) and Mathematics (15.04s), reflecting the additional computational effort required for complex quantitative problems. This pattern is consistent with Copilot's high detail rate on those subjects - more comprehensive answers take longer to generate.
Gemini - Fourth in speed (avg. 12.76s), close to Copilot. 
Gemini's response times are moderate and consistent across subjects, ranging from 10.62 seconds (Physics) to 16.81 seconds (Mathematical Science). Despite being the most accurate tool, Gemini does not incur the extreme time cost seen in ChatGPT, suggesting an efficient internal processing pipeline that balances thoroughness with speed.
ChatGPT - Slowest overall (avg. 15.78s). 
ChatGPT is consistently the slowest tool, with averages exceeding 19 seconds for both Mathematical Science and Mathematics. This is likely attributable to its high detailed response rate (88% across all subjects) - ChatGPT generates comprehensive, multi-paragraph explanations with worked steps, which demand more generation time. For quick lookup tasks, this latency is a significant user experience disadvantage.
 
5. Detailed Response Rate and Re-prompting Analysis
5.1 Detailed Response Rate by Tool
The ‘detailed response’ flag was recorded when the AI provided substantive explanatory content - including derivations, step-by-step solutions, contextual elaboration, or definitional depth - rather than a bare one-line answer. A high detailed response rate indicates that the tool adds explanatory value to academic queries, which is important in educational settings where understanding the reasoning process is as important as the final answer.

AI Tool	Physics	Math Sci.	Mathematics	Chemistry	Biotech
ChatGPT	40%	100%	100%	100%	100%
Copilot	40%	80%	100%	80%	100%
Perplexity	0%	0%	0%	0%	0%
Gemini	0%	20%	0%	60%	100%
Claude	0%	80%	60%	100%	60%

ChatGPT and Copilot dominate in providing detailed responses, particularly for quantitative subjects (Mathematics, Mathematical Science) and Biotechnology. This reflects their training optimisation toward comprehensive, tutorial-style outputs. Gemini and Claude provide detailed responses selectively, with Gemini doing so primarily in Biotechnology and Chemistry, and Claude in Chemistry and Mathematical Science. Perplexity provides zero detailed responses across all subjects - its output style is consistently brief and factual, resembling a search engine result snippet rather than an academic explanation.
5.2 Re-prompting Requirement
Re-prompting was required in a small minority of interactions. Only two tools registered any re-prompting requirement across the 25 questions each: Copilot (required re-prompting in 20% of Physics questions) and Gemini (required re-prompting in 20% of Physics questions). This means that for Mathematical Science, Mathematics, Chemistry, and Biotechnology, all five tools produced first-attempt responses that did not require follow-up rephrasing.
The Physics subject generated the only re-prompting instances - likely because Physics questions in this evaluation involved multi-step derivations or conceptual explanations that benefited from clarifying follow-up queries. All other tools (ChatGPT, Perplexity, Claude) required zero re-prompting across all 25 questions, indicating reliable first-attempt response quality.
 
6. Subject-wise Performance Summary
6.1 Physics
Physics was the only subject where all five tools except Perplexity achieved 100% accuracy. This suggests that Physics questions at this difficulty level are well within the competency envelope of current AI tools - particularly for ChatGPT, Copilot, Gemini, and Claude. Perplexity's 60% accuracy in Physics indicates difficulty with conceptual derivations or multi-step reasoning typical of university Physics problems. Response times in Physics were the lowest across all quantitative subjects, reflecting the relatively shorter output requirements of physics problem solutions compared to essay-style scientific subjects.
6.2 Mathematical Science
Mathematical Science proved to be a more discriminating domain. ChatGPT dropped to 80% accuracy (missing one question), and Perplexity fell to 40% - missing 3 of 5 questions. This subject generated the longest response times across all tools, with ChatGPT averaging 19.22 seconds, Copilot 17.57 seconds, and Gemini 16.81 seconds. The high detail rates for ChatGPT, Copilot, and Claude (100%, 80%, 80% respectively) reflect the necessity of worked mathematical derivations in this domain. Perplexity's failure to provide any detailed responses in Mathematical Science likely contributed directly to its low accuracy.
6.3 Mathematics
Mathematics produced the most consistent results across tools, with all tools except ChatGPT achieving 100% accuracy. ChatGPT's single incorrect response (question 5, time = 26.50s - the single longest recorded response time in the dataset) suggests a difficult or unusual problem that exceeded its reliable competency for that specific question. Perplexity's perfect accuracy in Mathematics, despite producing no detailed responses, indicates that for purely computational mathematical problems, concise numerical answers can be fully correct without elaboration. Response times in Mathematics were notably high for ChatGPT (19.53s avg.) and Copilot (15.04s avg.).
6.4 Chemistry
Chemistry was the most challenging subject for AI tools in this evaluation. Only Gemini achieved 100% accuracy. ChatGPT and Claude both scored just 40% (2 out of 5 correct), making Chemistry their weakest domain by a substantial margin. Copilot and Perplexity both managed 60%. The high detail rates of ChatGPT (100%) and Claude (100%) in Chemistry - despite their low accuracy - reveal an important finding: verbosity and detail do not guarantee correctness. Both tools produced comprehensive, well-structured Chemistry responses that were nonetheless factually incorrect, which is a significant risk in academic contexts where students may trust detailed-seeming AI responses without independent verification.
6.5 Biotechnology
Biotechnology produced the most balanced accuracy results, with ChatGPT achieving the sole perfect score (100%) and all other tools scoring 80%. This subject also generated the fastest average response times across the board (Perplexity: 4.43s, Copilot: 8.44s), suggesting that Biotechnology questions at this difficulty level are well-matched to current AI training data and require less extended reasoning chains. The high detail rates for ChatGPT, Copilot, and Gemini (all 100%) in Biotechnology suggest that all three tools provide not just correct but educationally rich responses in this domain.
 
7. Tool Rankings and Comprehensive Summary
7.1 Performance Rankings

Rank	By Accuracy	By Response Speed	By Detail Rate	By First-Attempt Reliability
1st	Gemini (96.0%)	Perplexity (6.67s)	ChatGPT (88.0%)	ChatGPT / Perplexity / Claude
2nd	Copilot (88.0%)	Claude (11.74s)	Copilot (80.0%)	- (0% re-prompt required)
3rd	Claude (84.0%)	Copilot (12.03s)	Claude (60.0%)	Copilot / Gemini (20% re-prompt in Physics)
4th	ChatGPT (80.0%)	Gemini (12.76s)	Gemini (36.0%)	-
5th	Perplexity (68.0%)	ChatGPT (15.78s)	Perplexity (0.0%)	-

7.2 Final Comprehensive Summary Table
The table below consolidates overall averages and provides a composite qualitative assessment of each AI tool’s suitability for academic use across the five dimensions measured in this evaluation.

AI Tool	Overall Accuracy	Avg. Response Time	Detailed Response Rate	Overall Assessment
ChatGPT	80%	15.78s	88%	Highly detailed but slowest. Good general accuracy; struggles in Chemistry.
Copilot	88%	12.03s	80%	Balanced: high accuracy + good detail. Occasionally needs re-prompting.
Perplexity	68%	6.67s	0%	Fastest by far but lowest accuracy. Best for quick factual lookups only.
Gemini	96%	12.76s	36%	Highest accuracy overall. Moderate speed. Detail coverage is inconsistent.
Claude	84%	11.74s	60%	Strong accuracy + fast. Best detail in STEM. Struggles in Chemistry.

8. Key Findings and Conclusions
Finding 1 - Gemini is the most accurate AI tool for university-level academic queries (96.0% overall). 
It achieved 100% accuracy in four of five subjects and was the only tool to score perfectly in Chemistry - the most discriminating subject domain in this evaluation. Gemini’s accuracy advantage is consistent across both conceptual and computational question types, making it the most reliable general-purpose academic tool among those tested.
Finding 2 - Perplexity is the fastest but least academically reliable (68.0% accuracy). 
The speed-accuracy trade-off for Perplexity is the sharpest of all tools. While it generates responses in under 7 seconds on average, it fails to provide detailed explanations (0% detail rate) and produces the lowest accuracy across STEM subjects. It is best suited to quick factual lookups rather than complex academic problem-solving.
Finding 3 - Detailed responses do not guarantee accuracy, particularly in Chemistry. 
ChatGPT and Claude both produced 100% detailed responses in Chemistry while achieving only 40% accuracy - the lowest accuracy of any tool-subject combination in this dataset. This is a critical finding for academic users: a comprehensive-seeming, well-structured AI response in a complex scientific domain may still be factually incorrect. Students should not use response verbosity or formatting quality as a proxy for correctness.
Finding 4 - Claude and Copilot offer the best accuracy-speed balance for academic use. 
Both tools achieve above-average accuracy (84% and 88% respectively) within competitive response time windows (11.74s and 12.03s respectively). For students requiring both correctness and reasonable response speed, these two tools represent the most practically efficient options for university-level academic assistance.
Finding 5 - Chemistry is the most challenging subject domain for current AI tools. 
Only Gemini achieved 100% accuracy in Chemistry. All other tools scored between 40–60%, indicating that current AI models have significant reliability limitations in advanced Chemistry - likely due to the precision required for molecular reasoning, reaction mechanisms, and chemical calculations, which are domains where small errors in intermediate steps compound into incorrect final answers.
Finding 6 - Re-prompting is rarely necessary, but Physics occasionally requires it. 
In 23 of 25 tool-subject combinations, zero re-prompting was required. The two exceptions (Copilot and Gemini in Physics) required follow-up prompts for 20% of their Physics questions. This suggests that current AI tools are generally capable of delivering first-attempt responses of sufficient quality for most academic domains, though Physics questions involving complex derivations may benefit from multi-turn dialogue.


Note: This evaluation is based on a structured sample of 5 questions per tool per subject (125 total observations). Results reflect performance on this specific question set and should be interpreted as indicative rather than definitive benchmarks. Tool accuracy and capability may vary with question phrasing, difficulty level, and version updates.
