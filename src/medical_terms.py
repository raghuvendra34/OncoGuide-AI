def simplify_terms(text: str):
    """
    ONLY extract medical terms and define them.
    NO summaries, NO report explanation.
    """

    prompt = f"""
Extract ONLY medical terms from the text and define them simply.

Return in format:
Term: Definition

TEXT:
{text}
"""

    from llm.llm_engine import generate_response

    return generate_response(prompt)