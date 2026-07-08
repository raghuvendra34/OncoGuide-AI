from datetime import datetime

from src.timeline_extractor import TimelineExtractor


class PatientTimeline:

    def __init__(self):
        self.events = []

    def add_report(self, report_text, report_type):

        dates = TimelineExtractor.extract_dates(report_text)

        event = TimelineExtractor.identify_event(report_text)

        if not dates:
            dates = ["Unknown Date"]

        for date in dates:

            new_event = {
    "date": date,
    "event": event,
    "report_type": report_type,
    "summary": report_text[:250]
    }

            if new_event not in self.events:
                self.events.append(new_event)

    def get_timeline(self):

        def parse_date(date):

            formats = [
                "%d/%m/%Y",
                "%d-%m-%Y",
                "%Y-%m-%d",
                "%d %B %Y",
            ]

            for fmt in formats:
                try:
                    return datetime.strptime(date, fmt)
                except:
                    pass

            return datetime.max

        return sorted(
            self.events,
            key=lambda x: parse_date(x["date"])
        )