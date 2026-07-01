def build_conversation_context(messages):
    """
    Convert recent conversation history into a readable context
    for the language model.
    """

    if not messages:
        return "No previous conversation."

    conversation = []

    for message in messages:

        role = message.get("role", "user").capitalize()

        content = message.get("content", "").strip()

        if not content:
            continue

        conversation.append(f"{role}: {content}")

    return "\n".join(conversation)