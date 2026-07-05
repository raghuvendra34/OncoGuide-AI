import re


IMPORTANT_HEADINGS = [
    "IMPRESSION",
    "DIAGNOSIS",
    "CONCLUSION",
    "FINDINGS",
    "OPINION",
    "SUMMARY",
    "ASSESSMENT",
    "RECOMMENDATION",
    "CLINICAL DETAILS",
    "HISTOPATHOLOGY",
    "MICROSCOPY",
    "FINAL DIAGNOSIS"
]


NOISE_PATTERNS = [
    r"NOT VALID FOR MEDICOLEGAL PURPOSE.*",
    r"Phone.*",
    r"WhatsApp.*",
    r"Registration No.*",
    r"ISO 9001.*",
    r"Department.*",
    r"Dr\..*",
    r"MD.*",
    r"MBBS.*",
    r"MS.*",
    r"Address.*",
    r"www\..*",
    r"http.*"
]


def remove_noise(text: str) -> str:
    """
    Remove obvious OCR noise and hospital information.
    """

    for pattern in NOISE_PATTERNS:
        text = re.sub(
            pattern,
            "",
            text,
            flags=re.IGNORECASE
        )

    return text


def remove_duplicate_lines(text: str) -> str:
    """
    Remove repeated OCR lines.
    """

    seen = set()
    cleaned = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if line in seen:
            continue

        seen.add(line)
        cleaned.append(line)

    return "\n".join(cleaned)


def prioritize_sections(text: str) -> str:
    """
    Move clinically important sections to the beginning.
    """

    important = []
    remaining = []

    lines = text.splitlines()

    for line in lines:

        upper = line.upper()

        if any(
            heading in upper
            for heading in IMPORTANT_HEADINGS
        ):
            important.append(line)

        else:
            remaining.append(line)

    return "\n".join(important + remaining)


def optimize_context(context: str) -> str:
    """
    Final optimization pipeline.
    """

    context = remove_noise(context)

    context = remove_duplicate_lines(context)

    context = prioritize_sections(context)

    return context.strip()