from ollama import chat


class PatientJourneyGenerator:
    """
    Generates a chronological patient journey using all uploaded reports.
    """

    def __init__(self, model="llama3"):
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

        for report in reports:

            report_context += f"""

Report Type:
{report['report_type']}

File:
{report['filename']}

Content:
{report['text'][:5000]}

"""

        prompt = f"""
You are an expert oncology medical assistant.

You are given multiple medical reports belonging to the SAME patient.

Create a chronological patient journey.

Instructions:

- Use only information present in the reports.
- Keep the language simple.
- Mention:
    • diagnosis
    • biopsy
    • surgeries
    • chemotherapy
    • radiation
    • scans
    • treatment response
    • follow-up

Return in this format:

Patient Journey

• ...

• ...

• ...

Do not invent information.

Reports:

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

            return response["message"]["content"]

        except Exception as e:

            return f"Unable to generate patient journey.\n\n{e}"