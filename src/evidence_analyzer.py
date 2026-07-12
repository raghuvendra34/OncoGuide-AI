def analyze_evidence(question: str, retrieved_chunks: list):
    """
    Analyze the quality of retrieved report evidence.

    Parameters:
        question (str): User's question.
        retrieved_chunks (list): Retrieved report text chunks.

    Returns:
        dict: Evidence analysis.
    """

    if not retrieved_chunks:
        return {
            "quality": "Low",
            "confidence": "Low",
            "reason": "No relevant report evidence was retrieved.",
            "missing_info": [
                "Relevant report information"
            ]
        }

    combined_text = " ".join(retrieved_chunks).strip()

    word_count = len(combined_text.split())

    if word_count > 300:
        quality = "High"
        confidence = "High"
        reason = "Multiple report sections support the answer."
    elif word_count > 120:
        quality = "Medium"
        confidence = "Medium"
        reason = "Some report evidence is available."
    else:
        quality = "Low"
        confidence = "Low"
        reason = "Limited report evidence was retrieved."

    return {
        "quality": quality,
        "confidence": confidence,
        "reason": reason,
        "missing_info": []
    }