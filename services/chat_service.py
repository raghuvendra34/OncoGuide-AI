from src.report_selector import detect_requested_report
from llm.prompt_builder import build_prompt
from llm.llm_engine import generate_response
from src.response_formatter import format_response
from src.evidence import clean_evidence
from src.conversation_context import build_conversation_context


def answer_question(question, context, memory):
    """
    PURE CHAT LAYER

    Uses report detection to guide the LLM toward
    the relevant report when multiple reports exist.
    """

    # -----------------------------
    # Clean retrieval context
    # -----------------------------
    context = clean_evidence(context)

    # -----------------------------
    # Memory
    # -----------------------------
    summary = memory.get_summary()

    recent = memory.get_recent_history(limit=4)

    conversation = build_conversation_context(recent)

    # -----------------------------
    # Detect requested report
    # -----------------------------
    requested_report = detect_requested_report(question)

    # -----------------------------
    # Guide the LLM
    # -----------------------------
    if requested_report:

        context = f"""
The user is specifically asking about the {requested_report.upper()} report.

Focus your answer on information related to that report.

If the requested report is not present in the retrieved context,
clearly tell the user instead of guessing.

Retrieved Context:

{context}
"""

    # -----------------------------
    # Build prompt
    # -----------------------------
    prompt = build_prompt(
        question=question,
        context=context,
        memory_summary=summary,
        conversation_context=conversation
    )

    # -----------------------------
    # Generate response
    # -----------------------------
    response = generate_response(prompt)

    formatted_response = format_response(response)

    return formatted_response, requested_report