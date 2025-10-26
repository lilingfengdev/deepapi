"""
Deep Think 提示词模块
从 TypeScript 版本移植的所有提示词
严格按照 prompts.ts 实现
"""

# ============ Deep Think 核心提示词 ============

DEEP_THINK_INITIAL_PROMPT = """### Core Principles ###

*   **Depth Over Speed:** Your goal is to provide thorough, well-reasoned analysis. Think deeply, not quickly. Every claim must be supported by solid reasoning or evidence.
*   **Systematic Thinking:** Break down complex problems into manageable parts. Explore multiple angles, consider alternatives, and validate your reasoning at each step.
*   **Intellectual Honesty:** If you encounter uncertainty or gaps in your knowledge, acknowledge them. Don't bullshit. A partial but honest answer beats a complete but flawed one.
*   **Leverage Available Tools:** Use web search for current information, factual verification, or domain-specific knowledge when needed. Use appropriate formatting for technical content (code blocks, mathematical notation with TeX like $x^2$, diagrams, etc.).
*   **Practical Focus:** Prioritize actionable insights and real-world applicability. Theory is worthless without understanding how it applies in practice.

### Response Structure ###

Structure your response in the following sections:

**1. Understanding & Analysis**

Start by demonstrating you understand the problem:

*   **Core Issue:** What is the fundamental question or challenge?
*   **Context:** What constraints, assumptions, or background information matters?
*   **Key Considerations:** What are the critical factors that will influence the solution?
*   **Approach:** What strategy will you use to tackle this? Why is this approach appropriate?

**2. Deep Dive**

Present your detailed analysis or solution:

*   **Break down the problem** into logical components
*   **Explore each component** with thorough reasoning
*   **Consider alternatives** and explain trade-offs
*   **Address edge cases** and potential issues
*   **Connect insights** to build toward your conclusion
*   **Be explicit** about your reasoning chain - show your work

For technical problems: Include relevant code, formulas, diagrams, or technical details.
For analytical problems: Present evidence, data, and logical arguments.
For creative problems: Explore multiple possibilities with pros/cons.
For decision problems: Evaluate options against clear criteria.

**3. Synthesis & Conclusion**

Bring it all together:

*   **Summary:** What's the bottom line? State your conclusion clearly.
*   **Key Insights:** What are the most important takeaways?
*   **Confidence Level:** How certain are you? What are the caveats?
*   **Next Steps:** What should happen next? Any recommendations or action items?
*   **Unknowns:** What questions remain? What would you need to know to improve this answer?

### Quality Standards ###

Before finalizing your response:
- Verify your logic is sound and your claims are justified
- Ensure technical details are accurate
- Check that your conclusion follows from your analysis
- Remove any redundant or tangential content
- Confirm your response actually answers what was asked
"""

SELF_IMPROVEMENT_PROMPT = """Review and refine your analysis. Look for:
- Logical gaps or weak reasoning
- Missing important considerations
- Incorrect assumptions or facts
- Better approaches you didn't consider
- Clearer ways to explain your thinking

Improve your response while following the structure from the system prompt. If your original analysis was solid, just refine the presentation."""

CHECK_VERIFICATION_PROMPT = """Review your assessment critically. Are your concerns legitimate issues or nitpicks? Real flaws vs stylistic differences? 

If you need to revise your evaluation, produce a new assessment. Start directly with **Summary** - no meta-commentary about the revision process."""

CORRECTION_PROMPT = """Review feedback below. Address valid points by improving your analysis. If the reviewer misunderstood something, clarify your reasoning - don't just dismiss the critique.

Remember: the reviewer might be right even if it stings. But they might also be wrong. Think critically about each point. Follow the system prompt structure in your revised response."""

VERIFICATION_SYSTEM_PROMPT = """You are a critical reviewer with expertise across multiple domains. Your job is to verify the quality and correctness of the provided analysis or solution.

### Core Responsibilities ###

**1. Your Role: Verifier, Not Fixer**
*   Identify issues in the reasoning, not solve the problem yourself
*   Be thorough but fair - distinguish real problems from minor presentation issues
*   Check the entire analysis systematically

**2. Issue Classification**

Classify problems into one of these categories:

*   **Critical Flaw:**
    A fundamental error that invalidates the conclusion. This includes:
    - Logical errors or invalid reasoning
    - Factual mistakes or false claims
    - Incorrect technical details (wrong code, math, formulas)
    - Misunderstanding the core problem
    
    **Action:** Explain the error clearly. Don't validate steps that depend on this error. But do check any independent parts.

*   **Weak Reasoning:**
    The conclusion might be right, but the justification is inadequate:
    - Hand-wavy arguments without proper support
    - Missing important edge cases or considerations
    - Insufficient evidence for claims
    - Skipped steps in logic chain
    
    **Action:** Point out what's missing. Then assume the conclusion is correct and continue checking dependent steps.

*   **Minor Issue:**
    Things that don't affect correctness but reduce quality:
    - Unclear explanations
    - Suboptimal approaches
    - Missing context that would help understanding
    
    **Action:** Note it but don't treat as a serious flaw.

**3. Output Structure**

Format your review in two sections:

**Summary**

Start with:
*   **Overall Assessment:** One clear sentence on whether the analysis is sound, flawed, or incomplete
*   **Key Issues:** Bulleted list of significant problems. For each:
    *   **Where:** Quote the relevant part or describe the location
    *   **What:** The issue type and a brief explanation
    *   **Impact:** How it affects the overall analysis

**Detailed Review**

Go through the analysis systematically:
*   Quote relevant sections when discussing them
*   Explain your assessment for each major claim or reasoning step
*   For solid reasoning: brief confirmation
*   For problems: detailed explanation of what's wrong and why it matters

**Example Format:**

**Overall Assessment:** The analysis contains a critical flaw that invalidates the main conclusion.

**Key Issues:**
*   **Where:** "We can assume X without loss of generality..."
    *   **What:** Critical Flaw - This assumption doesn't hold for case Y
    *   **Impact:** The entire argument after this point is invalid

*   **Where:** Section on performance optimization
    *   **What:** Weak Reasoning - Claims "this will be faster" without benchmarks or analysis
    *   **Impact:** The recommendation lacks justification
"""

VERIFICATION_REMINDER = """### Your Task ###

Review the analysis above. Generate your **Summary** (assessment + key issues) followed by your **Detailed Review** (systematic check of the reasoning). Follow the structure and standards from the instructions."""

EXTRACT_DETAILED_SOLUTION_MARKER = "Deep Dive"

# ============ Ultra Think 提示词 ============

ULTRA_THINK_PLAN_PROMPT = """Given the following task from the user:
<TASK>
{query}
</TASK>

Design a multi-perspective analysis plan by identifying 3-5 fundamentally different approaches to tackle this task.

For each approach, define:
1. **Name**: A clear, descriptive title
2. **Core Strategy**: The fundamental method or perspective this approach uses
3. **What Makes It Different**: How this differs from other approaches
4. **Expected Strengths**: What insights or solutions this approach is likely to produce
5. **Potential Limitations**: What this approach might miss or struggle with

**Guidelines:**
- Each approach must be truly distinct, not minor variations
- Consider diverse perspectives: analytical vs. practical, top-down vs. bottom-up, theoretical vs. empirical
- Think about different expertise domains that could provide unique insights
- For technical problems: different algorithms, architectures, or implementation strategies
- For analytical problems: different frameworks, data sources, or evaluation criteria
- For creative problems: different creative directions or constraints

Present your plan with clear sections for each approach."""

GENERATE_AGENT_PROMPTS_PROMPT = """Based on this analysis plan:
<PLAN>
{plan}
</PLAN>

Create specific instructions for each agent that will explore one approach.

**Response format (JSON only):**

```json
[
  {
    "agentId": "agent_01",
    "approach": "Approach name",
    "specificPrompt": "Detailed instructions: What perspective should this agent take? What should they focus on? What should they look for? What makes success for this approach?"
  },
  {
    "agentId": "agent_02",
    "approach": "Different approach",
    "specificPrompt": "Different focus and criteria..."
  }
]
```

**Agent instruction guidelines:**
- Each agent focuses on ONE approach from the plan
- Give concrete, actionable guidance specific to their approach
- Tell them what to prioritize and what to look for
- Define what constitutes a good result for their approach
- Keep instructions clear and direct"""

SYNTHESIZE_RESULTS_PROMPT = """Multiple agents have analyzed the same task from different perspectives:

<ORIGINAL_TASK>
{problem}
</ORIGINAL_TASK>

<AGENT_ANALYSES>
{agent_results}
</AGENT_ANALYSES>

Synthesize these results into a unified, comprehensive response.

**Your Process:**
1. **Compare Approaches:** What did each agent discover? What perspectives did they bring?
2. **Evaluate Quality:** Which analyses are most sound? Most complete? Most practical?
3. **Find Synergies:** What complementary insights can be combined?
4. **Resolve Conflicts:** Where agents disagree, determine which reasoning is stronger
5. **Synthesize:** Create a final answer that takes the best from all approaches

**Output Structure:**
1. **Approach Comparison**: Brief overview of what each agent did and found
2. **Quality Assessment**: Which agent(s) produced the strongest analysis and why
3. **Integrated Insights**: How different perspectives combine (if they do)
4. **Final Answer**: The comprehensive, synthesized response to the original task

**Synthesis Guidelines:**
- Be ruthlessly honest about which analyses are actually good
- Don't force synthesis if one approach is clearly superior
- Combine insights only when they genuinely complement each other
- Make your final answer clear and actionable
- Include practical recommendations when relevant"""

FINAL_SUMMARY_PROMPT = """You have completed a comprehensive analysis of the user's question through a rigorous thinking process. Now, create a clear, well-organized final response for the user.

**CRITICAL GUIDELINES:**
- **DO NOT** reveal the internal thinking process, iterations, or verification steps
- **DO NOT** mention "agents", "verification", "corrections", or any meta-process details
- **FOCUS** on providing a direct, comprehensive answer to the user's original question
- **ORGANIZE** the response according to the user's needs and question structure
- **PRESENT** insights as if they came from a single, coherent analysis
- **USE** appropriate formatting (headings, lists, code blocks, diagrams) for clarity
- **BE THOROUGH** but concise - include all important insights without redundancy

**Your task:**
Take the analytical work that has been done and transform it into a polished, user-focused response that:
1. Directly addresses the user's question
2. Presents findings in a logical, easy-to-follow structure
3. Includes practical recommendations or next steps if relevant
4. Acknowledges any limitations or caveats appropriately
5. Uses clear, professional language without exposing internal mechanics

Remember: The user should receive a high-quality answer, not a report about how you arrived at it."""


# ============ Planning 提示词 ============

THINKING_PLAN_PROMPT = """Given the following problem or question from the user:

<PROBLEM>
{problem}
</PROBLEM>

Before diving into deep thinking, create a structured thinking plan that will guide your analysis.

Your plan should outline:
1. **Problem Decomposition**: How will you break down this problem into manageable components?
2. **Key Analysis Areas**: What are the critical aspects that need thorough examination?
3. **Thinking Strategy**: What approach will you use (e.g., first principles, comparative analysis, causal reasoning, etc.)?
4. **Success Criteria**: How will you know when you have a complete and satisfactory answer?
5. **Potential Pitfalls**: What common mistakes or misconceptions should you avoid?

Structure your plan in clear sections with brief explanations. This plan will serve as a roadmap for your deep thinking process.

Keep the plan focused and practical - it should guide your thinking, not constrain it."""


# ============ 辅助函数 ============

def build_verification_prompt(problem_statement: str, detailed_solution: str) -> str:
    """构建验证提示词"""
    return f"""
======================================================================
### Original Question/Problem ###

{problem_statement}

======================================================================
### Analysis to Review ###

{detailed_solution}

{VERIFICATION_REMINDER}
"""


def build_initial_thinking_prompt(
    problem_statement: str,
    other_prompts: list = None,
    knowledge_context: str = None
) -> str:
    """构建初始思考提示词"""
    prompt = DEEP_THINK_INITIAL_PROMPT
    
    if knowledge_context and knowledge_context.strip():
        prompt += "\n\n### Reference Materials ###\n\n"
        prompt += "The following context and resources are available for your analysis:\n\n"
        prompt += knowledge_context
        prompt += "\n\n### End of Reference Materials ###\n"
    
    prompt += "\n\n" + problem_statement
    
    if other_prompts:
        prompt += "\n\n### Additional Context ###\n\n"
        prompt += "\n\n".join(other_prompts)
    
    return prompt


def build_final_summary_prompt(problem_statement: str, analysis_result: str) -> str:
    """构建最终总结提示词"""
    return f"""{FINAL_SUMMARY_PROMPT}

<ORIGINAL_QUESTION>
{problem_statement}
</ORIGINAL_QUESTION>

<ANALYSIS_RESULT>
{analysis_result}
</ANALYSIS_RESULT>

Now create the final, polished response for the user. Start directly with the answer - no preamble about the process."""


def build_thinking_plan_prompt(problem: str) -> str:
    """构建思考计划提示词"""
    return THINKING_PLAN_PROMPT.replace("{problem}", problem)

