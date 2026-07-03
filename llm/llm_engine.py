from ollama import chat


def generate_response(prompt):

    response = chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": "You are OncoGuide AI. Never hallucinate medical findings."
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response["message"]["content"]