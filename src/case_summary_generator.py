import json
import re
from ollama import chat


class CaseSummaryGenerator:

    def __init__(self, model="llama3"):
        self.model = model

    def generate_summary(
        self,
        extracted_information,
        patient_timeline,
    ):

        prompt = self._build_prompt(
            extracted_information,
            patient_timeline,
        )

        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        content = response["message"]["content"]

        return self._parse_response(content)

    def _build_prompt(
        self,
        extracted_information,
        patient_timeline,
    ):
        return f"""
You are an experienced oncology clinical assistant.

Use ONLY the provided information.

Structured Information:
{json.dumps(extracted_information, indent=2)}

Patient Timeline:
{json.dumps(patient_timeline, indent=2)}

Return ONLY valid JSON in this format:

{{
    "patient_information": "",
    "primary_diagnosis": "",
    "cancer_stage": "",
    "disease_progression": "",
    "treatments": "",
    "response_to_treatment": "",
    "current_status": "",
    "follow_up": "",
    "clinical_summary": "",
    "key_findings": []
}}
"""

    def _parse_response(self, response_text):

        try:
            match = re.search(
                r"\{.*\}",
                response_text,
                re.DOTALL
            )

            if match:
                return json.loads(match.group())

        except Exception:
            pass

        return {
            "patient_information": "Not Available",
            "primary_diagnosis": "Not Available",
            "cancer_stage": "Not Available",
            "disease_progression": "Not Available",
            "treatments": "Not Available",
            "response_to_treatment": "Not Available",
            "current_status": "Not Available",
            "follow_up": "Not Available",
            "clinical_summary": response_text,
            "key_findings": []
        }