def generate_patient_summary(text: str):
    """
    Generates a structured patient summary from all uploaded reports.
    """

    prompt = f"""
You are an experienced oncologist.

Using ONLY the medical information below, generate a structured patient summary.

MEDICAL REPORTS:
{text}

Return your answer in this format.

# Patient Summary

## Diagnosis
- Confirmed diagnosis

## Timeline
- Chronological medical events

## Treatments
- Surgery
- Chemotherapy
- Radiation
- Immunotherapy
- Medications

## Current Status
- Latest disease status
- Latest imaging/lab findings

## Key Findings
- Important clinical observations

Rules:
- Do not invent information.
- If information is missing, write "Not mentioned".
- Keep the summary concise.
- Use markdown headings and bullet points.
"""

    from llm.llm_engine import generate_response

    return generate_response(prompt)