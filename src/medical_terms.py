import re

from llm.llm_engine import generate_response


def simplify_terms(text: str):
    """
    Extract and explain only genuine medical terms found
    in the uploaded report.
    """

    # Reduce OCR noise
    cleaned_text = re.sub(r"\s+", " ", text)

    prompt = f"""
You are an oncology medical terminology assistant.

Your task is ONLY to identify genuine medical terms that are explicitly present
in the report and explain them in simple language.

STRICT RULES

- Use ONLY terms found in the report.
- Do NOT invent terms.
- Ignore OCR errors.
- Ignore hospital names.
- Ignore doctor names.
- Ignore patient names.
- Ignore registration numbers.
- Ignore dates.
- Ignore report headings.
- Ignore abbreviations that are unclear because of OCR.
- Do NOT explain invalid words.
- Remove duplicate terms.
- Maximum 15 terms.
- Keep each definition to one short sentence.

Return exactly in this format:

Medical Terms Explained

Term: Simple explanation

Term: Simple explanation

Medical Report:

{cleaned_text}
"""

    response = generate_response(prompt)

    return response.strip()