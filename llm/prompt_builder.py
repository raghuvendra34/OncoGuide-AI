from src.evidence import clean_evidence
from src.conversation_context import build_conversation_context


def build_prompt(question, context, memory):

    summary = memory.get_summary()
    recent_messages = memory.get_recent_history(limit=4)
    conversation = build_conversation_context(recent_messages)

    context = clean_evidence(context)

    return f"""
You are OncoGuide AI, a medical report assistant.

====================================================
SUMMARY
====================================================

{summary}

====================================================
CONVERSATION
====================================================

{conversation}

====================================================
REPORT CONTEXT
====================================================

{context}

====================================================
QUESTION
====================================================

{question}

====================================================
RULES
====================================================

- Use only report data if available
- Do not hallucinate
- Be simple and patient-friendly
- Separate report vs general knowledge

====================================================
ANSWER
====================================================
"""