def clean_evidence(context: str) -> str:
    """
    Clean retrieved report chunks before passing them to the LLM.
    """

    if not context:
        return ""

    # Split retrieved chunks
    chunks = context.split("\n\n")

    cleaned_chunks = []
    seen = set()

    for chunk in chunks:
        chunk = chunk.strip()

        if not chunk:
            continue

        # Remove duplicate chunks
        if chunk in seen:
            continue

        seen.add(chunk)

        # Limit very long chunks
        if len(chunk) > 700:
            chunk = chunk[:700] + "..."

        cleaned_chunks.append(chunk)

    return "\n\n".join(cleaned_chunks)