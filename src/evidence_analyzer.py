def analyze_evidence(question: str, retrieved_chunks: list):
    """
    Analyze the quality of retrieved report evidence.

    Parameters
    ----------
    question : str
        User question.

    retrieved_chunks : list
        Retrieved report chunks.

    Returns
    -------
    dict
        Evidence analysis.
    """

    if not retrieved_chunks:

        return {
            "quality": "Low",
            "confidence": 0,
            "reason": "No relevant report evidence was retrieved.",
            "missing_info": [
                "Relevant medical report"
            ]
        }

    chunk_count = len(retrieved_chunks)

    total_words = 0

    diagnosis_found = False
    treatment_found = False
    imaging_found = False

    keywords = [
        "diagnosis",
        "carcinoma",
        "cancer",
        "tumor",
        "lesion",
        "impression",
        "finding",
        "enhancing",
        "metastasis"
    ]

    for chunk in retrieved_chunks:

        text = chunk.lower()

        total_words += len(text.split())

        if any(word in text for word in [
            "diagnosis",
            "carcinoma",
            "cancer"
        ]):
            diagnosis_found = True

        if any(word in text for word in [
            "chemotherapy",
            "radiation",
            "surgery",
            "treatment"
        ]):
            treatment_found = True

        if any(word in text for word in [
            "mri",
            "ct",
            "pet",
            "scan",
            "impression",
            "lesion"
        ]):
            imaging_found = True

    score = 0

    # Number of retrieved chunks
    if chunk_count >= 3:
        score += 35
    elif chunk_count == 2:
        score += 25
    else:
        score += 15

    # Amount of evidence
    if total_words > 500:
        score += 30
    elif total_words > 250:
        score += 20
    elif total_words > 100:
        score += 10

    # Clinical evidence
    if diagnosis_found:
        score += 15

    if treatment_found:
        score += 10

    if imaging_found:
        score += 10

    score = min(score, 100)

    if score >= 85:
        quality = "Very High"

    elif score >= 70:
        quality = "High"

    elif score >= 50:
        quality = "Moderate"

    else:
        quality = "Low"

    reasons = []

    reasons.append(f"{chunk_count} report section(s) retrieved")

    if diagnosis_found:
        reasons.append("Diagnosis identified")

    if imaging_found:
        reasons.append("Imaging findings available")

    if treatment_found:
        reasons.append("Treatment information available")

    if total_words > 300:
        reasons.append("Rich supporting evidence")

    return {

        "quality": quality,

        "confidence": score,

        "reason": ". ".join(reasons),

        "missing_info": []
    }