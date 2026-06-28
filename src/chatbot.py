from ollama import chat


def answer_question(question, context, chat_history):
    """
    Answer user's question using:
    1. Uploaded medical report
    2. Retrieved report sections
    3. Previous conversation
    4. General medical knowledge when needed
    """

    # -----------------------------
    # Build Conversation History
    # -----------------------------
    conversation = ""

    if chat_history:

        recent_history = chat_history[-5:]

        for item in recent_history:

            conversation += (
                f"User: {item['question']}\n"
                f"Assistant: {item['answer']}\n\n"
            )


    # -----------------------------
    # Prompt
    # -----------------------------
    prompt = f"""
You are OncoGuide AI, an AI-powered educational cancer support assistant.

Your job is to help patients understand their uploaded medical reports
in simple and clear language.

====================================================
PREVIOUS CONVERSATION
====================================================

{conversation}


====================================================
RETRIEVED REPORT SECTIONS
====================================================

The first section below is the MOST RELEVANT section
for the user's question.

Use it as the PRIMARY source.

The remaining sections are supporting evidence.

Do not combine unrelated information.

{context}


====================================================
USER QUESTION
====================================================

{question}


====================================================
RULES
====================================================

1. Always prioritize the uploaded medical report.

2. Use previous conversation to understand follow-up questions.

3. If information exists in the report:

Start with:

## 📋 According to Your Report


4. If additional educational explanation is needed:

Create:

## 🩺 General Medical Information


Start that section with:

"The following information is based on general medical knowledge and is not specific to your uploaded report."


5. If information is not present:

Say:

"This information is not directly mentioned in the uploaded medical report."

Then provide general educational information.


6. Never:

- Diagnose diseases
- Prescribe medicines
- Recommend treatments
- Predict patient outcomes
- Invent medical findings


7. Keep answers:

- Simple
- Professional
- Patient-friendly
- Short paragraphs
- Bullet points when useful


8. Always finish with:

## ⚠ Disclaimer

This information is for educational purposes only and should not replace professional medical advice. Please consult your healthcare provider for medical guidance.


====================================================
ANSWER
====================================================
"""


    # -----------------------------
    # Ollama Response
    # -----------------------------
    response = chat(

        model="llama3",

        messages=[

            {
                "role": "system",
                "content":
                "You are OncoGuide AI, a professional cancer education assistant."
            },

            {
                "role": "user",
                "content": prompt
            }

        ]
    )


    return response["message"]["content"]