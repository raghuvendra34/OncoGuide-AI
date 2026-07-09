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

    # Helper to extract fields safely
    def get_field(field_name):
        """
        Extract a field from the LLM response.
        """

        start = extracted_text.find(field_name)

        if start == -1:
            return "Not Mentioned"

        start += len(field_name)

        fields = [
            "Diagnosis:",
            "Findings:",
            "Impression:",
            "Recommendations:",
            "Medications/Treatment:",
            "Abnormal Values:",
            "Key Medical Terms:"
        ]

        end = len(extracted_text)

        for f in fields:

            if f == field_name:
                continue

            pos = extracted_text.find(f, start)

            if pos != -1 and pos < end:
                end = pos

        return extracted_text[start:end].strip()


    medical_terms = get_field("Key Medical Terms:")

    medical_terms = [
        t.strip("-• ")
        for t in medical_terms.split("\n")
        if t.strip()
    ]


    return {

        "report_type": report_type,

        "diagnosis": get_field("Diagnosis:"),

        "findings": get_field("Findings:"),

        "impression": get_field("Impression:"),

        "recommendations": get_field("Recommendations:"),

        "treatment": get_field("Medications/Treatment:"),

        "abnormal_values": get_field("Abnormal Values:"),

        "medical_terms": medical_terms,

        "structured_summary": extracted_text
    }