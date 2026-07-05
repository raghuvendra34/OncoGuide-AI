def merge_chunks(chunks, max_chunks=5):
    """
    Merge top-ranked chunks into one coherent context.
    """

    selected = chunks[:max_chunks]

    merged = "\n\n----------------------\n\n".join(selected)

    return merged