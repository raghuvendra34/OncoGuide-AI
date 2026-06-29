from ollama import chat


def answer_question(question, context, memory):
    """
    Answer user's question using:
    1. Uploaded medical report
    2. Retrieved report sections
    3. Conversation summary
    4. Recent conversation memory
    5. General medical knowledge when needed
    """

    # -----------------------------
    # Get Memory Context
    # -----------------------------

    summary = memory.get_summary()

    recent_messages = memory.get_recent_history(
        limit=4
    )


    conversation = ""


    if recent_messages:

        for message in recent_messages:

            conversation += (
                f"{message['role'].capitalize()}: "
                f"{message['content']}\n\n"
            )


    # -----------------------------
    # Prompt
    # -----------------------------

    prompt = f"""
You are OncoGuide AI, an AI-powered educational cancer support assistant.

Your purpose is to help patients understand their uploaded medical reports
in simple, clear, and patient-friendly language.

====================================================
CONVERSATION SUMMARY
====================================================

{summary}


====================================================
RECENT CONVERSATION
====================================================

{conversation}


====================================================
RETRIEVED REPORT SECTIONS
====================================================

The first section is the MOST RELEVANT section for the user's question.

Use the uploaded report as the primary source.

Additional sections are supporting evidence.

Do not combine unrelated information.

{context}


====================================================
USER QUESTION
====================================================

{question}


====================================================
ANSWER RULES
====================================================

1. Always prioritize information from the uploaded medical report.

2. Use conversation memory to understand follow-up questions.

3. If the answer exists in the report, start with:

## 📋 According to Your Report


4. When explaining additional medical concepts, create:

## 🩺 General Medical Information


Start that section with:

"The following information is based on general medical knowledge and is not specific to your uploaded report."


5. If the information is not mentioned in the report:

Say:

"This information is not directly mentioned in the uploaded medical report."

Then provide general educational information.


6. Never:

- Diagnose diseases
- Prescribe medicines
- Recommend treatments
- Predict patient outcomes
- Invent medical findings


7. Do not mention the patient's name unless the user specifically asks.

Refer to it as:

"your report"

instead of using personal names.


8. Keep answers:

- Simple
- Professional
- Patient-friendly
- Short paragraphs
- Bullet points when useful


9. Never start answers with:

- "Let's address..."
- "Let's discuss..."
- "I will explain..."
- "Here is an overview..."
- "Let's understand..."

Start directly with the answer.


10. Always finish with:

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