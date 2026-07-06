from ollama import chat


def extract_medical_information(report_text: str, report_type: str):
    """
    Extract structured medical information from a medical report.

    Parameters
    ----------
    report_text : str
        Full extracted text from the report.

    report_type : str
        Report type detected by report_classifier.py

    Returns
    -------
    dict
        Structured medical information.
    """

    prompt = f"""
You are an expert medical information extraction system.

Your ONLY job is to extract information that is explicitly written in the report.

IMPORTANT RULES:
- Never guess.
- Never expand abbreviations.
- Never infer missing information.
- Never explain medical terms.
- Never create definitions.
- If something is unclear because of OCR errors, write "Unclear due to OCR".
- If a field is absent, write "Not Mentioned".

Report Type:
{report_type}

Medical Report:
{report_text}

Extract ONLY these fields:

Diagnosis:
Findings:
Impression:
Recommendations:
Medications/Treatment:
Abnormal Values:
Key Medical Terms:

Return ONLY the extracted fields.
"""

    response = chat(
        model="llama3:latest",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    extracted_text = response["message"]["content"].strip()

    return {
        "report_type": report_type,
        "structured_summary": extracted_text
    }