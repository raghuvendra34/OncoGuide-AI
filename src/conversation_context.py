def build_conversation_context(messages, max_messages=6):
    """
    Convert recent conversation history into a clean, readable context
    for the language model.

    Features:
    - Removes empty messages
    - Ignores duplicate consecutive messages
    - Labels User and Assistant clearly
    - Limits conversation length to reduce token usage
    """

    if not messages:
        return "No previous conversation."

    # Keep only the most recent messages
    messages = messages[-max_messages:]

    conversation = []
    previous_message = None

    for message in messages:

        role = message.get("role", "user").strip().lower()
        content = message.get("content", "").strip()

        if not content:
            continue

        # Skip duplicate consecutive messages
        if previous_message == (role, content):
            continue

        previous_message = (role, content)

        if role == "user":
            speaker = "User"
        elif role == "assistant":
            speaker = "Assistant"
        else:
            speaker = role.capitalize()

        conversation.append(f"{speaker}: {content}")

    if not conversation:
        return "No previous conversation."

    return (
        "Recent Conversation (oldest to newest):\n\n"
        + "\n".join(conversation)
    )