from ollama import chat


class PatientJourneyGenerator:
    """
    Generates a chronological patient journey using all uploaded reports.
    """

    def __init__(self, model="llama3:latest"):
        self.model = model

    def generate_journey(self, reports):
        """
        reports:
        [
            {
                "report_type": "...",
                "filename": "...",
                "text": "..."
            }
        ]
        """

        report_context = ""

        for i, report in enumerate(reports, start=1):

            report_context += f"""
==============================
REPORT {i}

Report Type:
{report['report_type']}

Filename:
{report['filename']}

Content:
{report['text'][:4000]}

"""

        prompt = f"""
You are an experienced oncology specialist.

You are reviewing multiple reports belonging to the SAME patient.

Your task is to reconstruct the patient's medical journey.

STRICT RULES

1. Use ONLY information explicitly written in the reports.
2. Never assume surgery, chemotherapy, radiation, biopsy or recurrence.
3. If a treatment is NOT mentioned, do NOT include it.
4. Ignore hospital names, doctor names, registration numbers and headers.
5. Keep events in chronological order whenever dates are available.
6. Keep each point short (1–2 sentences).
7. If information is unclear because of OCR, ignore it.
8. If only imaging reports are available, describe only the imaging findings.
9. Do NOT repeat the same finding multiple times.

Return EXACTLY in this format:

Patient Journey

• Initial diagnosis (if mentioned)

• Important imaging findings

• Treatments performed (only if explicitly stated)

• Follow-up findings

• Current disease status

If any section is unavailable, write:
Not Mentioned

Medical Reports:

{report_context}
"""

        try:

            response = chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response["message"]["content"].strip()

        except Exception as e:

            return f"Unable to generate patient journey.\n\n{e}"