import re


class TimelineExtractor:

    DATE_PATTERNS = [
        r"\b\d{2}/\d{2}/\d{4}\b",
        r"\b\d{2}-\d{2}-\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b\d{1,2}\s+[A-Za-z]+\s+\d{4}\b",
    ]

    EVENT_KEYWORDS = {
        "Diagnosis": ["diagnosis", "diagnosed", "confirmed", "pathology"],
        "Biopsy": ["biopsy"],
        "CT Scan": ["ct", "ct scan"],
        "PET Scan": ["pet", "pet ct"],
        "MRI": ["mri"],
        "Chemotherapy": ["chemotherapy", "cycle", "chemo"],
        "Radiation": ["radiation", "radiotherapy"],
        "Surgery": ["surgery", "operation", "resection"],
        "Follow-up": ["follow-up", "follow up", "review"],
        "Tumor Response": [
            "tumor reduced",
            "tumor increased",
            "stable disease",
            "partial response",
            "complete response",
            "progression",
        ],
    }

    @staticmethod
    def extract_dates(text):
        dates = []

        for pattern in TimelineExtractor.DATE_PATTERNS:
            dates.extend(re.findall(pattern, text))

        return list(set(dates))

    @staticmethod
    def identify_event(text):

        lower = text.lower()

        for event, keywords in TimelineExtractor.EVENT_KEYWORDS.items():

            for keyword in keywords:

                if keyword in lower:

                    sentences = text.replace("\n", " ").split(".")

                    for sentence in sentences:
                        if keyword in sentence.lower():
                            return sentence.strip()

                    return event

        return "Medical Event"