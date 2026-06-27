from ollama import chat


def answer_question(question, context, chat_history):
    """
    Answer the user's question using:
    1. Uploaded medical report
    2. Previous conversation
    3. General medical knowledge (only when needed)
    """

    # -----------------------------
    # Build Conversation History
    # -----------------------------
    conversation = ""

    if chat_history:
        # Keep only the last 5 conversations
        recent_history = chat_history[-5:]

        for chat_item in recent_history:
            conversation += (
                f"User: {chat_item['question']}\n"
                f"Assistant: {chat_item['answer']}\n\n"
            )

    # -----------------------------
    # Prompt
    # -----------------------------
    prompt = f"""
You are OncoGuide AI, an AI-powered educational cancer support assistant.

Your purpose is to help patients understand their uploaded medical reports
in simple, clear, and patient-friendly language.

====================================================
PREVIOUS CONVERSATION
====================================================

{conversation}

====================================================
MEDICAL REPORT
====================================================

{context}

====================================================
CURRENT QUESTION
====================================================

{question}

====================================================
INSTRUCTIONS
====================================================

1. The uploaded medical report is your PRIMARY source.

2. Use the previous conversation to understand follow-up questions.

Examples:
- "Explain that."
- "What does this mean?"
- "Tell me more."
- "What are its side effects?"
- "Can you summarize it?"

3. If the answer exists in the report:

Start with

## 📋 According to Your Report

Explain everything in simple language.

4. If additional explanation helps the patient:

Start another section:

## 🩺 General Medical Information

Begin this section with:

"The following information is based on general medical knowledge and is not specific to your uploaded report."

5. If the report does NOT contain the requested information:

Say:

"This information is not directly mentioned in the uploaded medical report."

Then provide general educational information.

6. Never:
- Diagnose diseases
- Recommend treatments
- Prescribe medicines
- Predict outcomes
- Invent report findings

7. Keep responses:
- Professional
- Patient-friendly
- Easy to understand
- Short paragraphs
- Bullet points whenever useful

8. Finish EVERY response with:

## ⚠ Disclaimer

This information is for educational purposes only and should not replace professional medical advice. Please consult your healthcare provider for medical guidance.

====================================================
ANSWER
====================================================
"""

    # -----------------------------
    # Call Ollama
    # -----------------------------
    response = chat(
        model="llama3",
        messages=[
            {
                "role": "system",
                "content": "You are OncoGuide AI, a professional educational cancer assistant."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]