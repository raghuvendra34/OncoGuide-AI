import re


class CaseSummaryGenerator:
    """
    Generates a clean cancer case summary from all structured reports.

    Improvements over the previous version:
    - does not rely only on the latest report
    - merges values across all reports
    - keeps the most informative non-empty value
    - preserves biomarkers, tumour details, and lab results
    - collects findings from every report
    """

    # Phrases that indicate a sentence is actually describing disease
    # status, even when it isn't filed under a literal "Current Status"
    # field (very often it's folded into Impression, Findings, or
    # Follow-up instead - e.g. "No evidence of disease recurrence. ECOG 0.").
    STATUS_KEYWORDS = [
        "no evidence of disease recurrence",
        "no evidence of recurrence",
        "no evidence of disease",
        "disease free",
        "disease-free",
        "stable disease",
        "progressive disease",
        "recurrence",
        "remission",
        "complete response",
        "partial response",
        "ecog",
        "performance status",
        "metastatic",
        "progression",
    ]

    # --------------------------------------------------
    # Basic cleaning helpers
    # --------------------------------------------------
    @staticmethod
    def _clean_value(value):
        """
        Normalise empty values.
        """
        if value is None:
            return "Not Mentioned"

        if isinstance(value, str):
            value = value.strip()
            if value == "" or value.lower() in ["not mentioned", "none", "unknown"]:
                return "Not Mentioned"
            return value

        if isinstance(value, list):
            return CaseSummaryGenerator._clean_list(value)

        if isinstance(value, dict):
            return value

        return value

    @staticmethod
    def _clean_list(values):
        """
        Normalise list fields.
        """
        if not values:
            return []

        if isinstance(values, str):
            values = [values]

        cleaned = []

        for value in values:
            value = str(value).strip()

            if (
                not value
                or value.lower() in ["not mentioned", "none", "unknown"]
            ):
                continue

            if value not in cleaned:
                cleaned.append(value)

        return cleaned

    @staticmethod
    def _is_empty(value):
        """
        Check whether a value is effectively empty.
        """
        if value is None:
            return True
        if value == "":
            return True
        if value == "Not Mentioned":
            return True
        if value == "Not mentioned":
            return True
        if value == {}:
            return True
        if value == []:
            return True
        return False

    @staticmethod
    def _report_priority(report_type):
        """
        Priority for choosing the best value.
        Higher number means more preferred.
        """
        rt = (report_type or "").lower()

        if any(word in rt for word in ["pathology", "biopsy", "histology", "histopathology"]):
            return 5
        if any(word in rt for word in ["lab", "blood", "hematology", "biochemistry"]):
            return 4
        if any(word in rt for word in ["mri", "ct", "scan", "imaging", "radiology", "x-ray", "pet"]):
            return 3
        if any(word in rt for word in ["follow", "discharge", "summary", "oncology", "review"]):
            return 2
        return 1

    @staticmethod
    def _value_score(value):
        """
        Heuristic score for choosing the more informative value.
        """
        if CaseSummaryGenerator._is_empty(value):
            return 0

        if isinstance(value, dict):
            score = 0
            for v in value.values():
                if not CaseSummaryGenerator._is_empty(v):
                    score += 1
            return score

        if isinstance(value, list):
            return len([v for v in value if not CaseSummaryGenerator._is_empty(v)])

        text = str(value).strip()
        score = len(text)

        if any(
            token in text.lower()
            for token in [
                "stage",
                "grade",
                "positive",
                "negative",
                "carcinoma",
                "adenocarcinoma",
                "biopsy",
                "mri",
                "ct",
                "scan",
                "surgery",
                "radiotherapy",
                "chemotherapy",
                "follow-up",
            ]
        ):
            score += 10

        return score

    @staticmethod
    def _choose_better_scalar(current_item, new_item):
        """
        Compare two scalar values with report priority and informativeness.
        Each item is a tuple: (value, report_type)
        """
        current_value, current_type = current_item
        new_value, new_type = new_item

        current_value = CaseSummaryGenerator._clean_value(current_value)
        new_value = CaseSummaryGenerator._clean_value(new_value)

        current_priority = CaseSummaryGenerator._report_priority(current_type)
        new_priority = CaseSummaryGenerator._report_priority(new_type)

        if CaseSummaryGenerator._is_empty(current_value) and not CaseSummaryGenerator._is_empty(new_value):
            return new_item

        if not CaseSummaryGenerator._is_empty(current_value) and CaseSummaryGenerator._is_empty(new_value):
            return current_item

        if new_priority > current_priority:
            return new_item

        if new_priority < current_priority:
            return current_item

        if CaseSummaryGenerator._value_score(new_value) > CaseSummaryGenerator._value_score(current_value):
            return new_item

        return current_item

    @staticmethod
    def _merge_list(current_list, new_list):
        """
        Merge list values without duplicates.
        """
        current_list = CaseSummaryGenerator._clean_list(current_list)
        new_list = CaseSummaryGenerator._clean_list(new_list)

        merged = []
        seen = set()

        for item in current_list + new_list:
            item = str(item).strip()
            if not item:
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)

        return merged

    @staticmethod
    def _merge_dict(current_dict, incoming_dict):
        """
        Merge nested dictionaries by keeping the more informative value
        for each key.
        """
        current_dict = current_dict or {}
        incoming_dict = incoming_dict or {}

        for key, value in incoming_dict.items():
            if key not in current_dict or CaseSummaryGenerator._is_empty(current_dict.get(key)):
                if not CaseSummaryGenerator._is_empty(value):
                    current_dict[key] = value
                continue

            current_value = current_dict.get(key)

            if isinstance(current_value, dict) and isinstance(value, dict):
                current_dict[key] = CaseSummaryGenerator._merge_dict(current_value, value)
            elif isinstance(current_value, list) and isinstance(value, list):
                current_dict[key] = CaseSummaryGenerator._merge_list(current_value, value)
            else:
                if CaseSummaryGenerator._choose_better_scalar(
                    (current_value, None),
                    (value, None),
                ) == (value, None):
                    current_dict[key] = value

        return current_dict

    @staticmethod
    def _infer_cancer_type(diagnosis):
        """
        Small heuristic to infer cancer type from diagnosis text.
        """
        text = str(diagnosis).lower()

        mapping = {
            "breast": "Breast Cancer",
            "lung": "Lung Cancer",
            "colon": "Colon Cancer",
            "rectal": "Rectal Cancer",
            "rectum": "Rectal Cancer",
            "prostate": "Prostate Cancer",
            "ovary": "Ovarian Cancer",
            "ovarian": "Ovarian Cancer",
            "cervix": "Cervical Cancer",
            "cervical": "Cervical Cancer",
            "stomach": "Gastric Cancer",
            "gastric": "Gastric Cancer",
            "liver": "Liver Cancer",
            "hepatic": "Liver Cancer",
            "pancreas": "Pancreatic Cancer",
            "pancreatic": "Pancreatic Cancer",
            "thyroid": "Thyroid Cancer",
            "kidney": "Kidney Cancer",
            "renal": "Kidney Cancer",
            "brain": "Brain Tumour",
            "glioma": "Brain Tumour",
            "lymphoma": "Lymphoma",
            "leukemia": "Leukaemia",
            "leukaemia": "Leukaemia",
            "melanoma": "Melanoma",
            "nasal": "Head and Neck Cancer",
            "sinus": "Head and Neck Cancer",
            "maxillary": "Head and Neck Cancer",
        }

        for key, value in mapping.items():
            if key in text:
                return value

        return "Not Mentioned"

    @staticmethod
    def _status_priority(report_type):
        """
        Priority for current-status extraction specifically. Follow-up
        and discharge reports are preferred here because they're the ones
        most likely to actually state where things stand today.
        """
        rt = (report_type or "").lower()

        if any(word in rt for word in ["follow", "discharge", "summary", "oncology", "review"]):
            return 5
        if any(word in rt for word in ["pathology", "biopsy", "histopathology", "histology"]):
            return 4
        if any(word in rt for word in ["mri", "ct", "scan", "imaging", "radiology", "x-ray", "pet"]):
            return 3
        if any(word in rt for word in ["lab", "blood", "hematology", "biochemistry"]):
            return 2
        return 1

    @staticmethod
    def _first_sentence(value):
        text = str(value).strip()
        if not text:
            return ""
        text = re.sub(r"\s+", " ", text)
        parts = re.split(r"(?<=[.!?])\s+", text)
        return parts[0].strip() if parts else text

    @staticmethod
    def _derive_current_status(reports):
        """
        A report rarely states status under a literal "Current Status"
        field - it's far more often folded into the Impression, Findings,
        or Follow-up text of the most recent follow-up/discharge report
        (e.g. "No evidence of disease recurrence. ECOG 0."). Falling back
        to scanning those fields for recognised status language means we
        don't report "Not Mentioned" when the report actually says so
        elsewhere.

        Returns (status_text, source_report_type).
        """
        best_fallback = ("Not Mentioned", None, -1)

        ordered_reports = sorted(
            reports,
            key=lambda r: CaseSummaryGenerator._status_priority(r.get("report_type")),
            reverse=True,
        )

        for report in ordered_reports:
            report_type = report.get("report_type") or "Not Mentioned"

            for field in ["current_status", "impression", "findings", "follow_up"]:
                raw = report.get(field)
                if CaseSummaryGenerator._is_empty(raw):
                    continue

                text = CaseSummaryGenerator._first_sentence(raw)
                if not text:
                    continue

                lowered = text.lower()
                if any(keyword in lowered for keyword in CaseSummaryGenerator.STATUS_KEYWORDS):
                    return text, report_type

                score = CaseSummaryGenerator._value_score(text) + CaseSummaryGenerator._status_priority(report_type)
                if score > best_fallback[2]:
                    best_fallback = (text, report_type, score)

        if best_fallback[0] != "Not Mentioned":
            return best_fallback[0], best_fallback[1]

        return "Not Mentioned", None

    @staticmethod
    def _pick_best_text_field(reports, field_name, allow_latest=False):
        """
        Choose the best scalar field across all reports.
        """
        best_value = "Not Mentioned"
        best_type = None

        for report in reports:
            value = report.get(field_name)
            if CaseSummaryGenerator._is_empty(value):
                continue

            report_type = report.get("report_type")
            candidate = CaseSummaryGenerator._choose_better_scalar(
                (best_value, best_type),
                (value, report_type),
            )

            best_value, best_type = candidate

        if allow_latest and best_value == "Not Mentioned" and reports:
            latest = reports[-1].get(field_name)
            if not CaseSummaryGenerator._is_empty(latest):
                best_value = latest

        return CaseSummaryGenerator._clean_value(best_value)

    @staticmethod
    def generate(medical_infos):
        """
        Generate a concise structured cancer case summary.
        """
        if not medical_infos:
            return {}

        summary = {
            "patient_information": {
                "name": "Not Mentioned",
                "age": "Not Mentioned",
                "gender": "Not Mentioned",
            },
            "primary_diagnosis": "Not Mentioned",
            "cancer_type": "Not Mentioned",
            "cancer_site": "Not Mentioned",
            "tumor_size": "Not Mentioned",
            "cancer_stage": "Not Mentioned",
            "histopathology": "Not Mentioned",
            "clinical_history": "Not Mentioned",
            "current_status": "Not Mentioned",
            "treatments": [],
            "medications": [],
            "follow_up": "Not Mentioned",
            "recommendations": [],
            "key_findings": [],
            "biomarkers": {},
            "tumor_details": {},
            "lab_results": {},
            "field_sources": {},
        }

        findings = []
        treatments = []
        medications = []
        recommendations = []

        biomarkers = {}
        tumor_details = {}
        lab_results = {}

        # --------------------------------------------------
        # Scalar fields: choose the most informative value
        # --------------------------------------------------
        summary["patient_information"]["name"] = CaseSummaryGenerator._pick_best_text_field(medical_infos, "patient_name")
        summary["patient_information"]["age"] = CaseSummaryGenerator._pick_best_text_field(medical_infos, "age")
        summary["patient_information"]["gender"] = CaseSummaryGenerator._pick_best_text_field(medical_infos, "gender")

        summary["primary_diagnosis"] = CaseSummaryGenerator._pick_best_text_field(medical_infos, "diagnosis")
        summary["cancer_type"] = CaseSummaryGenerator._pick_best_text_field(medical_infos, "cancer_type")

        if summary["cancer_type"] == "Not Mentioned" and summary["primary_diagnosis"] != "Not Mentioned":
            summary["cancer_type"] = CaseSummaryGenerator._infer_cancer_type(summary["primary_diagnosis"])

        summary["cancer_site"] = CaseSummaryGenerator._pick_best_text_field(medical_infos, "cancer_site")
        summary["tumor_size"] = CaseSummaryGenerator._pick_best_text_field(medical_infos, "tumor_size")
        summary["cancer_stage"] = CaseSummaryGenerator._pick_best_text_field(medical_infos, "cancer_stage")
        summary["histopathology"] = CaseSummaryGenerator._pick_best_text_field(medical_infos, "histopathology")
        summary["clinical_history"] = CaseSummaryGenerator._pick_best_text_field(medical_infos, "clinical_history")
        status_value, status_source = CaseSummaryGenerator._derive_current_status(medical_infos)
        summary["current_status"] = status_value
        # NOTE: field_sources is rebuilt from scratch further down, so
        # status_source is applied there (after that rebuild) instead of
        # here, where it would just get silently discarded.
        summary["follow_up"] = CaseSummaryGenerator._pick_best_text_field(medical_infos, "follow_up")

        # --------------------------------------------------
        # Lists
        # --------------------------------------------------
        for report in medical_infos:
            treatments = CaseSummaryGenerator._merge_list(treatments, report.get("treatments"))
            medications = CaseSummaryGenerator._merge_list(medications, report.get("medications"))
            recommendations = CaseSummaryGenerator._merge_list(recommendations, report.get("recommendations"))

        # A recommendation that just restates a treatment verbatim isn't
        # adding information - keep it only if it says something the
        # treatment list doesn't already say.
        treatment_keys = {t.strip().lower() for t in treatments}
        recommendations = [r for r in recommendations if r.strip().lower() not in treatment_keys]

        summary["treatments"] = treatments
        summary["medications"] = medications
        summary["recommendations"] = recommendations

        # --------------------------------------------------
        # Findings and impressions from all reports
        # --------------------------------------------------
        for report in medical_infos:
            for key in ["findings", "impression"]:
                value = report.get(key)
                if CaseSummaryGenerator._is_empty(value):
                    continue

                if isinstance(value, list):
                    for item in value:
                        item = str(item).strip()
                        if (
                            item
                            and item.lower() not in ["not mentioned", "none", "unknown"]
                            and item not in findings
                        ):
                            findings.append(item)
                else:
                    item = str(value).strip()
                    if (
                        item
                        and item.lower() not in ["not mentioned", "none", "unknown"]
                        and item not in findings
                    ):
                        findings.append(item)

        summary["key_findings"] = findings

        # --------------------------------------------------
        # Merge nested dictionaries
        # --------------------------------------------------
        for report in medical_infos:
            biomarkers = CaseSummaryGenerator._merge_dict(
                biomarkers,
                report.get("biomarkers", {}) or {},
            )
            tumor_details = CaseSummaryGenerator._merge_dict(
                tumor_details,
                report.get("tumor_details", {}) or {},
            )
            lab_results = CaseSummaryGenerator._merge_dict(
                lab_results,
                report.get("lab_results", {}) or {},
            )

        summary["biomarkers"] = biomarkers
        summary["tumor_details"] = tumor_details
        summary["lab_results"] = lab_results

        # If tumour details have a better stage, use it
        if summary["cancer_stage"] == "Not Mentioned" and tumor_details.get("stage"):
            summary["cancer_stage"] = CaseSummaryGenerator._clean_value(tumor_details.get("stage"))

        # If tumour details have a better size, use it
        if summary["tumor_size"] == "Not Mentioned" and tumor_details.get("size"):
            summary["tumor_size"] = CaseSummaryGenerator._clean_value(tumor_details.get("size"))

        # --------------------------------------------------
        # Field sources
        # --------------------------------------------------
        field_sources = {}

        for report in medical_infos:
            report_type = report.get("report_type") or "Not Mentioned"

            for field in [
                "patient_name",
                "age",
                "gender",
                "diagnosis",
                "cancer_type",
                "cancer_site",
                "tumor_size",
                "cancer_stage",
                "histopathology",
                "clinical_history",
                "current_status",
                "follow_up",
            ]:
                value = report.get(field)
                if not CaseSummaryGenerator._is_empty(value) and field not in field_sources:
                    field_sources[field] = report_type

            if report.get("biomarkers") and "biomarkers" not in field_sources:
                field_sources["biomarkers"] = report_type
            if report.get("tumor_details") and "tumor_details" not in field_sources:
                field_sources["tumor_details"] = report_type
            if report.get("lab_results") and "lab_results" not in field_sources:
                field_sources["lab_results"] = report_type
            if report.get("treatments") and "treatments" not in field_sources:
                field_sources["treatments"] = report_type
            if report.get("medications") and "medications" not in field_sources:
                field_sources["medications"] = report_type
            if report.get("recommendations") and "recommendations" not in field_sources:
                field_sources["recommendations"] = report_type

        if status_source:
            field_sources.setdefault("current_status", status_source)

        summary["field_sources"] = field_sources

        return summary