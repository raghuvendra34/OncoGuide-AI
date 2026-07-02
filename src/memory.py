from ollama import chat


class ConversationMemory:
    """
    Stores conversation history and maintains a rolling summary
    for long conversations.
    """

    def __init__(self):
        self.summary = ""
        self.chat_history = []

    # --------------------------------------------------
    # Add Message
    # --------------------------------------------------

    def add_message(self, role, content):
        """
        Add a message to conversation history.

        Duplicate consecutive messages are ignored.
        Automatically summarizes when the conversation becomes long.
        """

        content = content.strip()

        if not content:
            return

        # Prevent duplicate consecutive messages
        if (
            self.chat_history
            and self.chat_history[-1]["role"] == role
            and self.chat_history[-1]["content"] == content
        ):
            return

        self.chat_history.append(
            {
                "role": role,
                "content": content,
            }
        )

        # Automatically summarize after enough messages
        if len(self.chat_history) >= 10:
            self.summarize()

    # --------------------------------------------------
    # Get Recent History
    # --------------------------------------------------

    def get_recent_history(self, limit=4):
        """
        Return the most recent conversation messages.
        """

        return self.chat_history[-limit:]

    # --------------------------------------------------
    # Get Summary
    # --------------------------------------------------

    def get_summary(self):
        """
        Return the stored conversation summary.
        """

        return self.summary

    # --------------------------------------------------
    # Summarize Conversation
    # --------------------------------------------------

    def summarize(self):
        """
        Create or update a rolling summary of the conversation.
        """

        if len(self.chat_history) < 10:
            return

        conversation = ""

        for message in self.chat_history:
            role = message["role"].capitalize()
            content = message["content"]

            conversation += f"{role}: {content}\n"

        prompt = f"""
You are maintaining long-term memory for OncoGuide AI.

Your job is to update the conversation summary so future responses remain
consistent and avoid repeating previous explanations.

====================================================
PREVIOUS SUMMARY
====================================================

{self.summary}

====================================================
RECENT CONVERSATION
====================================================

{conversation}

====================================================
INSTRUCTIONS
====================================================

Update the summary while preserving important previous information.

Include ONLY:

• Important report findings discussed

• Medical concepts already explained

• Questions already answered

• Patient concerns

• Important follow-up topics

Do NOT include:

• Greetings

• Small talk

• Repeated information

• Formatting

Keep the summary concise (maximum 200 words).

Return only the updated summary.
"""

        try:

            response = chat(
                model="llama3",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You create concise medical conversation summaries "
                            "for OncoGuide AI."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            self.summary = response["message"]["content"].strip()

        except Exception:
            # Keep the previous summary if summarization fails
            pass

        # Keep only the newest messages after summarization
        self.chat_history = self.chat_history[-4:]

    # --------------------------------------------------
    # Clear Memory
    # --------------------------------------------------

    def clear(self):
        """
        Clear all stored memory.
        """

        self.summary = ""
        self.chat_history = []