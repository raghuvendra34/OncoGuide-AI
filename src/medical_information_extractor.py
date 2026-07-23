from ollama import chat
import re


def extract_medical_information(report_text: str, report_type: str):
    """
    Extract structured medical information from a medical report.
    """

    prompt = f"""
You are an expert medical information extraction system.

Your ONLY job is to extract information explicitly written in the report.

IMPORTANT RULES

- Never guess.
- Never infer missing information.
- Never expand abbreviations.
- Never explain medical terminology.
- Never create diagnoses that are not explicitly written.
- If OCR makes a field unreadable, write "Unclear due to OCR".
- If information is absent, write exactly "Not Mentioned".
- Ignore hospital names, page headers, page footers, doctor names, registration numbers, logos and addresses.
- Preserve dates exactly as written.

Extract ONLY these fields.

Patient Name:
Age:
Gender:

Report Type:
Report Date:

Diagnosis:
Cancer Type:
Cancer Site:
Tumor Size:
Cancer Stage:
Histopathology:

Clinical History:

Findings:

Impression:

Treatment:
Medications:

Follow-up:

Recommendations:

Abnormal Values:

Key Medical Terms:

OUTPUT RULES

- Return plain text only.
- Do NOT use Markdown.
- Do NOT use **bold**.
- Do NOT use bullet points unless multiple values exist.
- If a field is absent return exactly:
Not Mentioned

Medical Report:

{report_text}
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

    # ---------------------------------------
    # Cleanup LLM formatting
    # ---------------------------------------

    extracted_text = re.sub(r"\*\*", "", extracted_text)
    extracted_text = re.sub(r"__+", "", extracted_text)
    extracted_text = re.sub(r"`+", "", extracted_text)
    extracted_text = re.sub(r"\n{3,}", "\n\n", extracted_text)

    replacements = {
        "Not explicitly stated": "Not Mentioned",
        "None mentioned": "Not Mentioned",
        "None Mentioned": "Not Mentioned",
        "Unknown": "Not Mentioned",
    }

    for old, new in replacements.items():
        extracted_text = extracted_text.replace(old, new)

    extracted_text = extracted_text.strip()

    # ---------------------------------------
    # Field extractor
    # ---------------------------------------

    def get_field(field_name):

        start = extracted_text.find(field_name)

        if start == -1:
            return "Not Mentioned"

        start += len(field_name)

        fields = [

            "Patient Name:",
            "Age:",
            "Gender:",

            "Report Type:",
            "Report Date:",

            "Diagnosis:",
            "Cancer Type:",
            "Cancer Site:",
            "Tumor Size:",
            "Cancer Stage:",
            "Histopathology:",

            "Clinical History:",

            "Findings:",
            "Impression:",

            "Treatment:",
            "Medications:",

            "Follow-up:",

            "Recommendations:",

            "Abnormal Values:",

            "Key Medical Terms:"
        ]

        end = len(extracted_text)

        for field in fields:

            if field == field_name:
                continue

            pos = extracted_text.find(field, start)

            if pos != -1 and pos < end:
                end = pos

        value = extracted_text[start:end].strip()

        value = re.sub(r"^[\-\*\•\s]+", "", value)
        value = value.strip()

        normalized = value.lower()

        if normalized in [
            "",
            "not explicitly stated",
            "none mentioned",
            "none",
            "unknown",
        ]:
            return "Not Mentioned"

        return value

    # ---------------------------------------
    # Helper for list fields
    # ---------------------------------------

    def clean_list(value):

        if value == "Not Mentioned":
            return []

        items = []

        for line in value.split("\n"):

            line = line.strip("-•* ").strip()

            if not line:
                continue

            if line.lower() in [
                "not mentioned",
                "none mentioned",
                "none",
                "unknown",
            ]:
                continue

            items.append(line)

        return items

    # ---------------------------------------
    # Extract fields
    # ---------------------------------------

    medical_terms = clean_list(
        get_field("Key Medical Terms:")
    )

    treatments = clean_list(
        get_field("Treatment:")
    )

    medications = clean_list(
        get_field("Medications:")
    )

    recommendations = clean_list(
        get_field("Recommendations:")
    )

    # ---------------------------------------
    # Return structured information
    # ---------------------------------------

    return {

        # Patient
        "patient_name": get_field("Patient Name:"),
        "age": get_field("Age:"),
        "gender": get_field("Gender:"),

        # Report
        "report_type": report_type,
        "report_date": get_field("Report Date:"),

        # Diagnosis
        "diagnosis": get_field("Diagnosis:"),
        "cancer_type": get_field("Cancer Type:"),
        "cancer_site": get_field("Cancer Site:"),
        "tumor_size": get_field("Tumor Size:"),
        "cancer_stage": get_field("Cancer Stage:"),
        "histopathology": get_field("Histopathology:"),

        # Clinical
        "clinical_history": get_field("Clinical History:"),
        "findings": get_field("Findings:"),
        "impression": get_field("Impression:"),

        # Treatment
        "treatments": treatments,
        "medications": medications,
        "follow_up": get_field("Follow-up:"),

        # Other
        "recommendations": recommendations,
        "abnormal_values": get_field("Abnormal Values:"),
        "medical_terms": medical_terms,

        # Raw LLM output
        "structured_summary": extracted_text
    }