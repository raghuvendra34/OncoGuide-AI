def format_response(response: str) -> str:
    """
    Clean and standardize the LLM response before displaying it.
    """

    if not response:
        return "No response generated."

    response = response.strip()

    # Remove repeated headings
    duplicates = [
        "###  Report Information\n###  Report Information",
        "###  General Medical Information\n###  General Medical Information",
        "###  Evidence from Report\n###  Evidence from Report",
        "###  Disclaimer\n###  Disclaimer",
    ]

    for item in duplicates:
        parts = item.split("\n")
        response = response.replace(item, parts[0])

    # Remove excessive blank lines
    while "\n\n\n" in response:
        response = response.replace("\n\n\n", "\n\n")

    return response.strip()