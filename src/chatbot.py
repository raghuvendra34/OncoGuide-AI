from llm.llm_engine import generate_response
from src.response_formatter import format_response
from src.evidence import clean_evidence
from src.conversation_context import build_conversation_context


# --------------------------------------------------
# Build Prompt (SEPARATED)
# --------------------------------------------------

def build_prompt(question, context, memory):
    summary = memory.get_summary()
    recent_messages = memory.get_recent_history(limit=4)
    conversation = build_conversation_context(recent_messages)

    context = clean_evidence(context)

    prompt = f"""
You are OncoGuide AI, an educational AI assistant that helps patients understand their uploaded cancer-related medical reports.

====================================================
CONVERSATION SUMMARY
====================================================

{summary}

====================================================
RECENT CONVERSATION
====================================================

{conversation}

====================================================
RETRIEVED REPORT SECTIONS
====================================================

{context}

====================================================
USER QUESTION
====================================================

{question}

====================================================
RESPONSE RULES
====================================================

- Use ONLY report data if available.
- If missing, say it's not in report.
- Do NOT hallucinate medical findings.
- Separate report info and general knowledge clearly.
- Keep language simple and patient-friendly.

====================================================
OUTPUT FORMAT
====================================================

### Report Information

### General Medical Information (if needed)

### Evidence from Report

### Disclaimer (if needed)

====================================================
ANSWER
====================================================
"""
    return prompt

# --------------------------------------------------
# MAIN FUNCTION (CLEAN)
# --------------------------------------------------

def answer_question(question, context, memory):

    prompt = build_prompt(question, context, memory)

    answer = generate_response(prompt)

    answer = format_response(answer)

    return answer