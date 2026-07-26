from src.report_selector import detect_requested_report
from llm.prompt_builder import build_prompt
from llm.llm_engine import generate_response
from src.response_formatter import format_response
from src.evidence import clean_evidence
from src.conversation_context import build_conversation_context
from src.context_optimizer import optimize_context


def build_patient_summary_context(patient_summary):
    """
    Converts the structured patient summary into text
    that can be injected into the LLM prompt.
    """

    if not patient_summary:
        return ""

    summary = f"""
=========================
PATIENT MEDICAL SUMMARY
=========================

Patient Name:
{patient_summary.get("patient_name", "Unknown")}

Diagnosis:
{patient_summary.get("diagnosis", "Unknown")}

Cancer Stage:
{patient_summary.get("cancer_stage", "Unknown")}

Treatments:
{", ".join(patient_summary.get("treatments", [])) or "None"}

Medications:
{", ".join(patient_summary.get("medications", [])) or "None"}

Tumor Details:
{patient_summary.get("tumor_details", {})}

Biomarkers:
{patient_summary.get("biomarkers", {})}

Lab Results:
{patient_summary.get("lab_results", {})}

Recommendations:
{", ".join(patient_summary.get("recommendations", [])) or "None"}

Timeline:
"""

    for event in patient_summary.get("timeline", []):

        summary += f"""

• Report Type : {event.get("report_type", "Unknown")}
  Report Date : {event.get("report_date", "Unknown")}
  Diagnosis   : {event.get("diagnosis", "Unknown")}
  Stage       : {event.get("stage", "Unknown")}

"""

    return summary


def answer_question(
    question,
    context,
    memory,
    patient_summary=None
):
    """
    Main RAG chatbot function.

    Uses:
    - Cross-report patient summary
    - Conversation memory
    - Retrieved report evidence
    """

    # --------------------------------------------------
    # Compatibility with new retrieval_service
    # --------------------------------------------------

    if isinstance(context, dict):
        context = context.get("context", "")

    if context is None:
        context = ""

    # --------------------------------------------------
    # Clean retrieved evidence
    # --------------------------------------------------

    context = clean_evidence(context)
    context = optimize_context(context)

    # --------------------------------------------------
    # Conversation memory
    # --------------------------------------------------

    summary = memory.get_summary()

    if summary is None:
        summary = ""

    if isinstance(summary, dict):
        summary = str(summary)

    recent = memory.get_recent_history(limit=8)

    conversation = build_conversation_context(recent)

    if conversation is None:
        conversation = ""

    # --------------------------------------------------
    # Detect requested report
    # --------------------------------------------------

    requested_report = detect_requested_report(question)

    # --------------------------------------------------
    # Patient summary
    # --------------------------------------------------

    patient_context = build_patient_summary_context(patient_summary)

    # --------------------------------------------------
    # Final context
    # --------------------------------------------------

    final_context = f"""
{patient_context}

====================================
RETRIEVED REPORT EVIDENCE
====================================

{context}
"""

    # --------------------------------------------------
    # Focus on requested report
    # --------------------------------------------------

    if requested_report:

        final_context = f"""
The user is asking specifically about the
{requested_report.upper()} report.

Focus on that report.

If it is unavailable,
say so instead of guessing.

{final_context}
"""

    # --------------------------------------------------
    # Build Prompt
    # --------------------------------------------------

    prompt = build_prompt(
        question=question,
        context=final_context,
        memory_summary=summary,
        conversation_context=conversation,
        requested_report=requested_report
    )

    # --------------------------------------------------
    # Generate Response
    # --------------------------------------------------

    response = generate_response(prompt)

    # --------------------------------------------------
    # Format Response
    # --------------------------------------------------

    formatted = format_response(response)

    return formatted, requested_report