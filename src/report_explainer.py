from ollama import chat


def explain_report(report_text):

    prompt = f"""
You are OncoGuide AI, an assistant that helps patients understand medical reports.

IMPORTANT INSTRUCTIONS:

- The report text may have been extracted using OCR.
- OCR can introduce spelling mistakes, formatting issues, and missing words.
- Carefully infer the intended medical meaning.
- Do not assume information that is not clearly present.
- If a finding is unclear, mention that the report text may contain OCR errors.
- Explain everything in simple patient-friendly language.
- Do NOT provide a diagnosis.
- Do NOT predict survival chances.
- Do NOT guarantee treatment outcomes.
- Clearly state that this is not medical advice.

Provide the response in this format:

## Simple Summary

## Important Findings

## Medical Terms Explained

## Possible Next Steps

## Questions for the Doctor

## Disclaimer

Report Text:
{report_text[:8000]}
"""

    response = chat(
        model="llama3",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]