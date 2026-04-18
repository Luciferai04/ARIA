# aria/prompts/contract_prompt.py
# Anthropic Harness Pattern: Sprint Contract
# Negotiates what "done" looks like before the answer is generated.

CONTRACT_PROMPT = """\
You are the Application Evaluator for ARIA. 
Acting in a "Sprint Contract" pattern, your job is to define EXACTLY what the Generator must produce for its answer to be considered successful, based ONLY on the retrieved context.

--- QUESTION ---
{question}

--- PLANNED SUB-QUERIES ---
{sub_queries}

--- RETRIEVED CONTEXT ---
{context}

--- INSTRUCTIONS ---
Write a structured "Answer Contract" outlining the criteria for a successful response.
Specifically detail:
1. REQUIRED COVERAGE: What specific facts/points from the context MUST be included to fully answer the question?
2. REQUIRED SOURCES: What sources from the context MUST be cited?
3. SAFETY CONSTRAINTS: Explicitly state if the context lacks information to fully answer the question (to prevent hallucination). If the user's question contains a false premise not supported by the context, explicitly mandate that the premise MUST be corrected in the answer.

Respond ONLY with valid JSON (no markdown, no extra keys):
{{
  "required_coverage": ["point 1", "point 2"],
  "required_sources": ["source 1", "source 2"],
  "safety_constraints": "Clear instruction regarding false premise or hallucination avoidance"
}}
"""
