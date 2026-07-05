import re


HIGH_PRIORITY = [
    "IMPRESSION",
    "DIAGNOSIS",
    "FINAL DIAGNOSIS",
    "CONCLUSION",
    "FINDINGS",
    "ASSESSMENT",
    "RECOMMENDATION",
    "CLINICAL IMPRESSION"
]


MEDIUM_PRIORITY = [
    "HISTORY",
    "OBSERVATION",
    "EXAMINATION",
    "RESULT",
    "MICROSCOPY",
    "GROSS",
    "OPINION"
]


LOW_PRIORITY = [
    "PHONE",
    "WHATSAPP",
    "ADDRESS",
    "ISO",
    "REGISTRATION",
    "FAX",
    "EMAIL",
    "DR.",
    "MD",
    "MBBS"
]


def score_chunk(chunk: str):

    score = 0

    upper = chunk.upper()

    for word in HIGH_PRIORITY:
        if word in upper:
            score += 100

    for word in MEDIUM_PRIORITY:
        if word in upper:
            score += 40

    for word in LOW_PRIORITY:
        if word in upper:
            score -= 60

    score += min(len(chunk) // 40, 30)

    return score


def rank_chunks(chunks):

    scored = []

    for chunk in chunks:
        scored.append(
            (
                score_chunk(chunk),
                chunk
            )
        )

    scored.sort(reverse=True)

    return [c for _, c in scored]