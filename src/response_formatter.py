import re


def format_response(response: str) -> str:
    """
    Clean and standardize the LLM response before displaying it.

    Features:
    - Removes duplicate headings
    - Removes duplicate disclaimer sections
    - Removes excessive blank lines
    - Normalizes heading formatting
    - Removes trailing spaces
    """

    if not response or not response.strip():
        return "No response generated."

    response = response.strip()

    # ----------------------------------------
    # Normalize heading spacing
    # ----------------------------------------

    heading_map = {
        "### Report Information": "### Report Information",
        "###  Report Information": "### Report Information",

        "### General Medical Information": "### General Medical Information",
        "###  General Medical Information": "### General Medical Information",

        "### Evidence from Report": "### Evidence from Report",
        "###  Evidence from Report": "### Evidence from Report",

        "### Disclaimer": "### Disclaimer",
        "###  Disclaimer": "### Disclaimer",
    }

    for old, new in heading_map.items():
        response = response.replace(old, new)

    # ----------------------------------------
    # Remove duplicate consecutive headings
    # ----------------------------------------

    headings = [
        "### Report Information",
        "### General Medical Information",
        "### Evidence from Report",
        "### Disclaimer",
    ]

    for heading in headings:
        duplicate = f"{heading}\n{heading}"
        while duplicate in response:
            response = response.replace(duplicate, heading)

    # ----------------------------------------
    # Remove duplicate disclaimer section
    # ----------------------------------------

    disclaimer = (
        "This information is for educational purposes only "
        "and should not replace professional medical advice. "
        "Please consult your healthcare provider for medical guidance."
    )

    if response.count(disclaimer) > 1:
        first = response.find(disclaimer)
        last = first + len(disclaimer)

        response = (
            response[:last]
            + response[last:].replace(disclaimer, "")
        )

    # ----------------------------------------
    # Remove trailing whitespace
    # ----------------------------------------

    lines = [line.rstrip() for line in response.splitlines()]

    response = "\n".join(lines)

    # ----------------------------------------
    # Remove excessive blank lines
    # ----------------------------------------

    response = re.sub(r"\n{3,}", "\n\n", response)

    # ----------------------------------------
    # Final cleanup
    # ----------------------------------------

    return response.strip()