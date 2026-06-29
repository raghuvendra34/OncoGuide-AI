from ollama import chat


class ConversationMemory:
    def __init__(self):
        self.summary = ""
        self.chat_history = []

    def add_message(self, role, content):
        """
        Add a message to the conversation history.
        """
        self.chat_history.append({
            "role": role,
            "content": content
        })

    def get_recent_history(self, limit=4):
        """
        Return the last few messages for context.
        """
        return self.chat_history[-limit:]

    def get_summary(self):
        """
        Return the stored conversation summary.
        """
        return self.summary

    def summarize(self):
        """
        Summarize the conversation if it becomes too long.
        """

        if len(self.chat_history) < 10:
            return

        conversation = ""

        for msg in self.chat_history:
            conversation += f"{msg['role'].capitalize()}: {msg['content']}\n"

        prompt = f"""
You are summarizing a conversation between a cancer patient and an AI assistant.

Focus only on:

- Patient concerns
- Questions already answered
- Important report findings discussed
- Medical explanations already given

Keep the summary concise.

Conversation:

{conversation}
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

        self.summary = response["message"]["content"]

        # Keep only the most recent messages
        self.chat_history = self.chat_history[-4:]