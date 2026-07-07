from collections import defaultdict


class CrossReportReasoner:
    """
    Combines structured information extracted from
    multiple medical reports into one unified summary.
    """

    def __init__(self, extracted_reports):
        """
        extracted_reports:
        List of dictionaries returned by the
        Medical Information Extraction Engine.
        """
        self.reports = extracted_reports

    def build_summary(self):

        summary = {
            "patient_name": None,
            "diagnosis": None,
            "cancer_stage": None,
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
        labs = {}
        tumor = {}

        for report in self.reports:

            if not summary["patient_name"]:
                summary["patient_name"] = report.get("patient_name")

            if not summary["diagnosis"]:
                summary["diagnosis"] = report.get("diagnosis")

            if not summary["cancer_stage"]:
                summary["cancer_stage"] = report.get("cancer_stage")

            treatments.update(report.get("treatments", []))
            medications.update(report.get("medications", []))
            recommendations.update(report.get("recommendations", []))

            biomarkers.update(report.get("biomarkers", {}))
            labs.update(report.get("lab_results", {}))
            tumor.update(report.get("tumor_details", {}))

            summary["timeline"].append({
                "report_type": report.get("report_type"),
                "report_date": report.get("report_date"),
                "diagnosis": report.get("diagnosis"),
                "stage": report.get("cancer_stage")
            })

        summary["treatments"] = sorted(list(treatments))
        summary["medications"] = sorted(list(medications))
        summary["recommendations"] = sorted(list(recommendations))

        summary["biomarkers"] = biomarkers
        summary["lab_results"] = labs
        summary["tumor_details"] = tumor

        return summary