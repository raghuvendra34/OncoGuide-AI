from ollama import chat


def answer_question(question, context):

    prompt = f"""
You are OncoGuide AI.

Answer the user's question using the report context below.

Rules:
- Use the report context as the primary source.
- If additional general medical knowledge is used, clearly say:
  "Based on general medical knowledge..."
- If the answer is not present in the report, say:
  "This information is not directly mentioned in the uploaded report."
- Do not invent patient-specific details.
- Do not diagnose.
- Do not predict outcomes.
- Keep answers simple and patient-friendly.
- End with: "Please consult your doctor for medical advice."

Report Context:
{context}

Question:
{question}
"""

    response = chat(
        model="llama3",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]