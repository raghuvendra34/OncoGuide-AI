class CrossReportReasoner:
    """
    Combines structured information extracted from
    multiple medical reports into one unified summary.
    """

    def __init__(self, extracted_reports):
        self.reports = extracted_reports

    def normalize_list(self, value):
        """
        Ensure every list field is actually a clean list.
        """

        if value is None:
            return []

        if isinstance(value, list):
            return [
                item.strip()
                for item in value
                if item and item.strip() and item != "Not Mentioned"
            ]

        if isinstance(value, str):

            value = value.strip()

            if (
                not value
                or value.lower() == "not mentioned"
                or value.lower() == "none"
            ):
                return []

            return [value]

        return []

    def build_summary(self):

        summary = {
            "patient_name": "Not Mentioned",
            "diagnosis": "Not Mentioned",
            "cancer_stage": "Not Mentioned",
            "treatments": [],
            "medications": [],
            "biomarkers": {},
            "tumor_details": {},
            "lab_results": {},
            "recommendations": [],
            "timeline": []
        }

        treatments = set()
        medications = set()
        recommendations = set()

        biomarkers = {}
        tumor = {}
        labs = {}

        for report in self.reports:

            if (
                summary["patient_name"] == "Not Mentioned"
                and report.get("patient_name")
            ):
                summary["patient_name"] = report["patient_name"]

            if (
                summary["diagnosis"] == "Not Mentioned"
                and report.get("diagnosis")
            ):
                summary["diagnosis"] = report["diagnosis"]

            if (
                summary["cancer_stage"] == "Not Mentioned"
                and report.get("cancer_stage")
            ):
                summary["cancer_stage"] = report["cancer_stage"]

            treatments.update(
                self.normalize_list(
                    report.get("treatments")
                )
            )

            medications.update(
                self.normalize_list(
                    report.get("medications")
                )
            )

            recommendations.update(
                self.normalize_list(
                    report.get("recommendations")
                )
            )

            biomarkers.update(
                report.get("biomarkers", {})
            )

            tumor.update(
                report.get("tumor_details", {})
            )

            labs.update(
                report.get("lab_results", {})
            )

            summary["timeline"].append({
                "report_type": report.get("report_type"),
                "report_date": report.get("report_date"),
                "diagnosis": report.get("diagnosis"),
                "stage": report.get("cancer_stage")
            })

        summary["treatments"] = sorted(treatments)
        summary["medications"] = sorted(medications)
        summary["recommendations"] = sorted(recommendations)

        summary["biomarkers"] = biomarkers
        summary["tumor_details"] = tumor
        summary["lab_results"] = labs

        return summary