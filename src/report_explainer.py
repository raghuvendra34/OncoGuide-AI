def explain_report(text: str):
    """
    ONLY generates high-level summary.
    NO medical definitions inside this module.
    """

    prompt = f"""
You are a medical report summarizer.

Summarize the report clearly:

- Diagnosis
- Key findings
- Treatment plan
- Patient status

Do NOT explain medical terms.
Do NOT repeat definitions.

REPORT:
{text}
"""

    from llm.llm_engine import generate_response

    return generate_response(prompt)