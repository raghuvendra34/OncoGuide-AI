import re


def detect_report_type(text):
    """
    Detect the type of uploaded medical report.
    """

    text = text.lower()

    blood_keywords = [
        "hemoglobin",
        "wbc",
        "rbc",
        "platelet",
        "cbc",
        "blood",
        "hematology",
        "neutrophils",
        "lymphocytes",
        "esr"
    ]

    biopsy_keywords = [
        "biopsy",
        "histopathology",
        "tissue",
        "specimen",
        "core biopsy",
        "needle biopsy"
    ]

    pathology_keywords = [
        "pathology",
        "microscopic",
        "gross examination",
        "diagnosis",
        "histological",
        "malignant",
        "benign"
    ]

    imaging_keywords = [
        "ct scan",
        "mri",
        "pet scan",
        "ultrasound",
        "x-ray",
        "radiology",
        "contrast"
    ]

    discharge_keywords = [
        "discharge summary",
        "admitted",
        "hospital stay",
        "follow-up",
        "discharged"
    ]

    prescription_keywords = [
        "tablet",
        "capsule",
        "medicine",
        "dosage",
        "prescription",
        "take once daily"
    ]

    categories = {
        "Blood Test Report": blood_keywords,
        "Biopsy Report": biopsy_keywords,
        "Pathology Report": pathology_keywords,
        "Imaging Report": imaging_keywords,
        "Discharge Summary": discharge_keywords,
        "Prescription": prescription_keywords,
    }

    scores = {}

    for report_type, keywords in categories.items():
        score = 0

        for keyword in keywords:
            score += len(re.findall(re.escape(keyword), text))

        scores[report_type] = score

    best_match = max(scores, key=scores.get)

    if scores[best_match] == 0:
        return "General Medical Report"

    return best_match