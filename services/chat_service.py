from llm.prompt_builder import build_prompt
from llm.llm_engine import generate_response
from src.response_formatter import format_response
from src.evidence import clean_evidence
from src.conversation_context import build_conversation_context


def answer_question(question, context, memory):
    """
    PURE CHAT LAYER (NO EXTRA LOGIC)
    Only responsible for Q&A.
    """

    # Clean retrieval context
    context = clean_evidence(context)

    # Memory
    summary = memory.get_summary()
    recent = memory.get_recent_history(limit=4)
    conversation = build_conversation_context(recent)

    # Build prompt
    prompt = build_prompt(
        question=question,
        context=context,
        memory_summary=summary,
        conversation_context=conversation
    )

    # LLM call
    response = generate_response(prompt)

    return format_response(response)