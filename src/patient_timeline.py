from datetime import datetime
from src.timeline_extractor import TimelineExtractor


class PatientTimeline:

    def __init__(self):
        self.events = []

    def add_report(
        self,
        report_text,
        report_type,
        structured_info=None
    ):
        """
        Add one report to the patient timeline.

        If structured medical information is available,
        prefer it over raw OCR text.
        """

        dates = TimelineExtractor.extract_dates(report_text)

        if not dates:
            if structured_info:
                report_date = structured_info.get("report_date")

                if report_date not in [
                    "",
                    None,
                    "Not Mentioned"
                ]:
                    dates = [report_date]

        if not dates:
            dates = ["Unknown Date"]

        if structured_info:

            diagnosis = structured_info.get(
                "diagnosis",
                "Not Mentioned"
            )

            findings = structured_info.get(
                "findings",
                "Not Mentioned"
            )

            impression = structured_info.get(
                "impression",
                "Not Mentioned"
            )

            event = ""

            if diagnosis != "Not Mentioned":
                event += f"Diagnosis: {diagnosis}\n"

            if findings != "Not Mentioned":
                event += f"Findings: {findings}\n"

            if impression != "Not Mentioned":
                event += f"Impression: {impression}"

            if event.strip() == "":
                event = "Clinical information not available."

            summary = impression

        else:

            event = TimelineExtractor.identify_event(report_text)

            summary = report_text[:200]

        for date in dates:

            timeline_event = {

                "date": date,

                "event": event.strip(),

                "report_type": report_type,

                "summary": summary
            }

            if timeline_event not in self.events:
                self.events.append(timeline_event)

    def get_timeline(self):

        def parse_date(date):

            formats = [

                "%d/%m/%Y",

                "%d/%m/%y",

                "%d-%m-%Y",

                "%Y-%m-%d",

                "%d %B %Y",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(date, fmt)
                except Exception:
                    continue

            return datetime.max

        return sorted(
            self.events,
            key=lambda x: parse_date(x["date"])
        )