import re


class CrossReportReasoner:
    """
    Combines structured information extracted from multiple medical reports
    into one unified summary.

    This version is priority-aware:
    - pathology reports are preferred for diagnosis, histopathology, biomarkers
    - imaging reports are preferred for tumour size, site, and stage clues
    - laboratory reports are preferred for lab values
    - follow-up/discharge reports are preferred for treatments, medications,
      current status, and follow-up plans
    """

    def __init__(self, extracted_reports):
        self.reports = extracted_reports or []

    # --------------------------------------------------
    # Normalisation helpers
    # --------------------------------------------------
    def normalize_list(self, value):
        """
        Ensure every list field is returned as a clean list.
        """
        if value is None:
            return []

        if isinstance(value, list):
            cleaned = []
            seen = set()
            for item in value:
                if item is None:
                    continue
                item = str(item).strip()
                if not item:
                    continue
                if item.lower() in {"not mentioned", "not mentioned.", "none", "unknown"}:
                    continue
                key = item.lower()
                if key in seen:
                    continue
                seen.add(key)
                cleaned.append(item)
            return cleaned

        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.lower() in {"not mentioned", "none", "unknown"}:
                return []
            return [value]

        return []

    def clean_value(self, value):
        """
        Normalise a scalar value for comparisons and display.
        """
        if value is None:
            return "Not Mentioned"

        if isinstance(value, dict):
            return value

        if isinstance(value, list):
            cleaned = self.normalize_list(value)
            return cleaned if cleaned else "Not Mentioned"

        value = str(value).strip()
        if not value:
            return "Not Mentioned"

        lowered = value.lower()
        if lowered in {
            "not mentioned",
            "not mentioned.",
            "none",
            "none mentioned",
            "unknown",
            "not explicitly stated",
        }:
            return "Not Mentioned"

        return value

    def is_empty(self, value):
        """
        Return True if a value is effectively empty.
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

    def value_score(self, value):
        """
        Heuristic score for choosing the better value.
        Higher score means more informative.
        """
        if self.is_empty(value):
            return 0

        if isinstance(value, dict):
            score = 0
            for v in value.values():
                if not self.is_empty(v):
                    score += 1
            return score

        if isinstance(value, list):
            return len([v for v in value if not self.is_empty(v)])

        text = str(value).strip()
        score = len(text)

        # Mild bonus for medically informative values
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
                "ultrasound",
                "surgery",
                "radiotherapy",
                "chemotherapy",
                "follow-up",
            ]
        ):
            score += 10

        return score

    def merge_scalar(self, current_value, new_value):
        """
        Keep the more informative non-empty scalar value.
        """
        current_value = self.clean_value(current_value)
        new_value = self.clean_value(new_value)

        if self.is_empty(current_value) and not self.is_empty(new_value):
            return new_value

        if not self.is_empty(current_value) and self.is_empty(new_value):
            return current_value

        if self.value_score(new_value) > self.value_score(current_value):
            return new_value

        return current_value

    def merge_list(self, current_list, new_list):
        """
        Merge list values without duplicates.
        """
        current_list = self.normalize_list(current_list)
        new_list = self.normalize_list(new_list)

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

    def merge_dict(self, current_dict, incoming_dict):
        """
        Merge nested dictionaries by keeping the more informative value
        for each key.
        """
        current_dict = current_dict or {}
        incoming_dict = incoming_dict or {}

        for key, value in incoming_dict.items():
            if key not in current_dict or self.is_empty(current_dict.get(key)):
                if not self.is_empty(value):
                    current_dict[key] = value
                continue

            current_value = current_dict.get(key)

            if isinstance(current_value, dict) and isinstance(value, dict):
                current_dict[key] = self.merge_dict(current_value, value)
            elif isinstance(current_value, list) and isinstance(value, list):
                current_dict[key] = self.merge_list(current_value, value)
            else:
                current_dict[key] = self.merge_scalar(current_value, value)

        return current_dict

    # --------------------------------------------------
    # Report priority
    # --------------------------------------------------
    def report_priority(self, report_type):
        """
        Return a numeric priority for choosing better values.
        Higher means more preferred for certain fields.
        """
        rt = (report_type or "").lower()

        if any(word in rt for word in ["pathology", "biopsy", "histopathology", "histology"]):
            return 5

        if any(word in rt for word in ["lab", "blood", "hematology", "biochemistry"]):
            return 4

        if any(word in rt for word in ["mri", "ct", "scan", "imaging", "radiology", "x-ray", "pet"]):
            return 3

        if any(word in rt for word in ["follow", "discharge", "summary", "oncology", "review"]):
            return 2

        return 1

    def status_priority(self, report_type):
        """
        Priority for current status extraction.
        Follow-up/discharge reports are preferred here.
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

    def choose_better_by_priority(self, current_item, new_item):
        """
        Compare two values using report priority and informativeness.
        Each item is expected to be a tuple:
            (value, report_type)
        """
        current_value, current_type = current_item
        new_value, new_type = new_item

        current_value = self.clean_value(current_value)
        new_value = self.clean_value(new_value)

        current_priority = self.report_priority(current_type)
        new_priority = self.report_priority(new_type)

        if self.is_empty(current_value) and not self.is_empty(new_value):
            return new_item

        if not self.is_empty(current_value) and self.is_empty(new_value):
            return current_item

        if new_priority > current_priority:
            return new_item

        if new_priority < current_priority:
            return current_item

        if self.value_score(new_value) > self.value_score(current_value):
            return new_item

        return current_item

    # --------------------------------------------------
    # Text helpers
    # --------------------------------------------------
    def clean_text(self, value):
        """
        Clean scalar text for status extraction and deduplication.
        """
        if value is None:
            return ""

        if isinstance(value, list):
            value = " ".join(str(v).strip() for v in value if v)

        elif isinstance(value, dict):
            parts = []
            for k, v in value.items():
                if v is None or v == "" or v == "Not Mentioned":
                    continue
                parts.append(f"{k}: {v}")
            value = " ".join(parts)

        else:
            value = str(value)

        value = value.strip()
        value = re.sub(r"\s+", " ", value)
        value = re.sub(
            r"^(diagnosis|findings|impression|current status|status|follow-up|follow up|treatment|recommendations)\s*:\s*",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = value.strip(" -")
        return value.strip()

    def trim_to_sentence(self, text):
        """
        Return the first sentence if the text is multi-sentence.
        """
        text = self.clean_text(text)
        if not text:
            return "Not Mentioned"

        parts = re.split(r"(?<=[.!?])\s+", text)
        first = parts[0].strip() if parts else text.strip()
        return first if first else "Not Mentioned"

    def normalize_finding(self, text):
        """
        Normalise findings/impression strings for deduplication.
        """
        text = self.trim_to_sentence(text)
        if not text or text == "Not Mentioned":
            return "Not Mentioned"
        return text

    def derive_current_status(self):
        """
        Derive a single current status from all reports.
        Returns (status_text, source_report_type).
        """
        status_keywords = [
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

        best_fallback = ("Not Mentioned", None, -1)

        ordered_reports = sorted(
            self.reports,
            key=lambda r: self.status_priority(r.get("report_type")),
            reverse=True,
        )

        for report in ordered_reports:
            report_type = report.get("report_type") or "Not Mentioned"

            for field in ["current_status", "impression", "findings", "follow_up"]:
                raw = report.get(field)
                if self.is_empty(raw):
                    continue

                text = self.trim_to_sentence(raw)
                if not text or text == "Not Mentioned":
                    continue

                lowered = text.lower()
                if any(keyword in lowered for keyword in status_keywords):
                    return text, report_type

                score = self.value_score(text) + self.status_priority(report_type)
                if score > best_fallback[2]:
                    best_fallback = (text, report_type, score)

        if best_fallback[0] != "Not Mentioned":
            return best_fallback[0], best_fallback[1]

        return "Not Mentioned", None

    # --------------------------------------------------
    # Main summary builder
    # --------------------------------------------------
    def build_summary(self):
        summary = {
            "patient_name": "Not Mentioned",
            "diagnosis": "Not Mentioned",
            "cancer_type": "Not Mentioned",
            "cancer_site": "Not Mentioned",
            "cancer_stage": "Not Mentioned",
            "histopathology": "Not Mentioned",
            "clinical_history": "Not Mentioned",
            "current_status": "Not Mentioned",
            "key_findings": [],
            "treatments": [],
            "medications": [],
            "biomarkers": {},
            "tumor_details": {},
            "lab_results": {},
            "recommendations": [],
            "timeline": [],
            "field_sources": {}
        }

        # Track best scalar values together with their source report type
        best_values = {
            "patient_name": ("Not Mentioned", None),
            "diagnosis": ("Not Mentioned", None),
            "cancer_type": ("Not Mentioned", None),
            "cancer_site": ("Not Mentioned", None),
            "cancer_stage": ("Not Mentioned", None),
            "histopathology": ("Not Mentioned", None),
            "clinical_history": ("Not Mentioned", None),
        }

        treatments = []
        medications = []
        recommendations = []
        key_findings = []

        biomarkers = {}
        tumor = {}
        labs = {}

        for report in self.reports:
            report_type = report.get("report_type")
            report_date = report.get("report_date")

            # --------------------------------------------------
            # Core scalar fields
            # --------------------------------------------------
            for field in [
                "patient_name",
                "diagnosis",
                "cancer_type",
                "cancer_site",
                "cancer_stage",
                "histopathology",
                "clinical_history",
            ]:
                incoming_value = report.get(field)
                if incoming_value is None:
                    continue

                best_values[field] = self.choose_better_by_priority(
                    best_values[field],
                    (incoming_value, report_type),
                )

            # --------------------------------------------------
            # Lists
            # --------------------------------------------------
            treatments = self.merge_list(treatments, report.get("treatments"))
            medications = self.merge_list(medications, report.get("medications"))
            recommendations = self.merge_list(recommendations, report.get("recommendations"))

            # --------------------------------------------------
            # Key findings
            # --------------------------------------------------
            findings = report.get("findings")
            impression = report.get("impression")

            if findings and not self.is_empty(findings):
                if isinstance(findings, list):
                    for item in findings:
                        norm = self.normalize_finding(item)
                        if norm != "Not Mentioned" and norm.lower() not in [f.lower() for f in key_findings]:
                            key_findings.append(norm)
                else:
                    norm = self.normalize_finding(findings)
                    if norm != "Not Mentioned" and norm.lower() not in [f.lower() for f in key_findings]:
                        key_findings.append(norm)

            if impression and not self.is_empty(impression):
                if isinstance(impression, list):
                    for item in impression:
                        norm = self.normalize_finding(item)
                        if norm != "Not Mentioned" and norm.lower() not in [f.lower() for f in key_findings]:
                            key_findings.append(norm)
                else:
                    norm = self.normalize_finding(impression)
                    if norm != "Not Mentioned" and norm.lower() not in [f.lower() for f in key_findings]:
                        key_findings.append(norm)

            # --------------------------------------------------
            # Nested dictionaries
            # --------------------------------------------------
            biomarkers = self.merge_dict(biomarkers, report.get("biomarkers", {}))
            tumor = self.merge_dict(tumor, report.get("tumor_details", {}))
            labs = self.merge_dict(labs, report.get("lab_results", {}))

            # --------------------------------------------------
            # Timeline
            # --------------------------------------------------
            timeline_event = {
                "date": report_date or "Not Mentioned",
                "event": self.build_event_text(report),
                "report_type": report_type or "Not Mentioned",
            }
            summary["timeline"].append(timeline_event)

            # --------------------------------------------------
            # Field sources
            # --------------------------------------------------
            self.update_field_sources(summary["field_sources"], report)

        # Finalise scalar fields
        for field, (value, source) in best_values.items():
            summary[field] = self.clean_value(value)
            if source:
                summary["field_sources"][field] = source

        # A recommendation that just restates a treatment verbatim isn't
        # adding information - keep it only if it says something the
        # treatment list doesn't already say.
        treatment_keys = {t.strip().lower() for t in treatments}
        recommendations = [r for r in recommendations if r.strip().lower() not in treatment_keys]

        # Finalise lists
        summary["treatments"] = treatments
        summary["medications"] = medications
        summary["recommendations"] = recommendations

        # Finalise findings
        summary["key_findings"] = key_findings

        # Finalise nested dictionaries
        summary["biomarkers"] = biomarkers
        summary["tumor_details"] = tumor
        summary["lab_results"] = labs

        # Derive current status from status-bearing report text
        current_status, current_status_source = self.derive_current_status()
        summary["current_status"] = current_status
        if current_status_source:
            summary["field_sources"]["current_status"] = current_status_source

        # Prefer nested tumor fields as the canonical stage if available
        if self.is_empty(summary["cancer_stage"]) and tumor.get("stage"):
            summary["cancer_stage"] = tumor["stage"]

        # Improve cancer type from diagnosis when explicit cancer type is missing
        if self.is_empty(summary["cancer_type"]) and summary["diagnosis"] != "Not Mentioned":
            summary["cancer_type"] = self.infer_cancer_type(summary["diagnosis"])

        # Keep timeline order stable
        summary["timeline"] = self.sort_timeline(summary["timeline"])

        return summary

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def build_event_text(self, report):
        parts = []

        diagnosis = report.get("diagnosis")
        if diagnosis and not self.is_empty(diagnosis):
            parts.append(f"Diagnosis: {self.trim_to_sentence(diagnosis)}")

        findings = report.get("findings")
        if findings and not self.is_empty(findings):
            findings_text = self.trim_to_sentence(findings)
            if findings_text:
                parts.append(f"Findings: {findings_text}")

        impression = report.get("impression")
        if impression and not self.is_empty(impression):
            impression_text = self.trim_to_sentence(impression)
            findings_text = self.trim_to_sentence(findings) if findings and not self.is_empty(findings) else ""
            if impression_text and impression_text != findings_text:
                parts.append(f"Impression: {impression_text}")

        treatment = report.get("treatments")
        if treatment and not self.is_empty(treatment):
            if isinstance(treatment, list):
                parts.append(f"Treatment: {', '.join(self.normalize_list(treatment))}")
            else:
                parts.append(f"Treatment: {self.trim_to_sentence(treatment)}")

        follow_up = report.get("follow_up")
        if follow_up and not self.is_empty(follow_up):
            parts.append(f"Follow-up: {self.trim_to_sentence(follow_up)}")

        if not parts:
            return "Medical report processed"

        return " ".join(parts)

    def update_field_sources(self, field_sources, report):
        """
        Store the likely source report type for each field when present.
        """
        report_type = report.get("report_type") or "Not Mentioned"

        for field in [
            "diagnosis",
            "cancer_type",
            "cancer_site",
            "cancer_stage",
            "histopathology",
            "clinical_history",
        ]:
            value = report.get(field)
            if value and not self.is_empty(value):
                existing = field_sources.get(field)
                if not existing:
                    field_sources[field] = report_type

        if report.get("current_status"):
            field_sources.setdefault("current_status", report_type)

        if report.get("biomarkers"):
            field_sources.setdefault("biomarkers", report_type)

        if report.get("tumor_details"):
            field_sources.setdefault("tumor_details", report_type)

        if report.get("lab_results"):
            field_sources.setdefault("lab_results", report_type)

        if report.get("treatments"):
            field_sources.setdefault("treatments", report_type)

        if report.get("medications"):
            field_sources.setdefault("medications", report_type)

        if report.get("recommendations"):
            field_sources.setdefault("recommendations", report_type)

    def infer_cancer_type(self, diagnosis):
        """
        Very small heuristic to infer cancer type from diagnosis text.
        This is only used when an explicit cancer type is absent.
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

    def sort_timeline(self, timeline):
        """
        Keep timeline stable.
        """
        if not timeline:
            return []

        return timeline