"""
Report Selector

This module detects which medical report the user is referring to
(e.g., Biopsy, Blood Test, MRI, CT Scan) based on their question.

If no report is mentioned, it returns None so the chatbot can
search across all uploaded reports.
"""

import re


REPORT_KEYWORDS = {
    "biopsy": [
        "biopsy",
        "histopathology",
        "pathology",
        "tissue",
    ],
    "blood": [
        "blood",
        "cbc",
        "hemoglobin",
        "haemoglobin",
        "wbc",
        "rbc",
        "platelet",
    ],
    "mri": [
        "mri",
        "magnetic resonance",
    ],
    "ct": [
        "ct",
        "ct scan",
        "computed tomography",
    ],
    "pet": [
        "pet",
        "pet scan",
    ],
    "xray": [
        "xray",
        "x-ray",
    ],
    "ultrasound": [
        "ultrasound",
        "sonography",
    ],
}


def detect_requested_report(question: str):
    """
    Detect which report the user is asking about.

    Parameters
    ----------
    question : str
        User's question.

    Returns
    -------
    str | None
        Returns report type (e.g. "biopsy", "blood")
        or None if no report is detected.
    """

    question = question.lower().strip()

    for report_type, keywords in REPORT_KEYWORDS.items():
        for keyword in keywords:
            if re.search(rf"\b{re.escape(keyword)}\b", question):
                return report_type

    return None