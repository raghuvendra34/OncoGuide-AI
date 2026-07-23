class CaseSummaryGenerator:
    """
    Generates a clean cancer case summary from all structured reports.
    """

    @staticmethod
    def _clean_value(value):
        """
        Normalize empty values.
        """

        if value is None:
            return "Not Mentioned"

        if isinstance(value, str):
            value = value.strip()

            if (
                value == ""
                or value.lower() in [
                    "not mentioned",
                    "none",
                    "unknown",
                ]
            ):
                return "Not Mentioned"

            return value

        return value

    @staticmethod
    def _clean_list(values):
        """
        Normalize list fields.
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
                or value.lower() in [
                    "not mentioned",
                    "none",
                    "unknown",
                ]
            ):
                continue

            if value not in cleaned:
                cleaned.append(value)

        return cleaned

    @staticmethod
    def generate(medical_infos):
        """
        Generate a concise structured cancer case summary.
        """

        if not medical_infos:
            return {}

        latest = medical_infos[-1]

        findings = []

        for report in medical_infos:

            finding = report.get("findings", "").strip()

            if finding not in [
                "",
                "Not Mentioned",
                "Unclear due to OCR",
            ] and finding not in findings:

                findings.append(finding)

        summary = {

            # Patient
            "patient_information": {
                "name": CaseSummaryGenerator._clean_value(
                    latest.get("patient_name")
                ),
                "age": CaseSummaryGenerator._clean_value(
                    latest.get("age")
                ),
                "gender": CaseSummaryGenerator._clean_value(
                    latest.get("gender")
                ),
            },

            # Diagnosis
            "primary_diagnosis":
                CaseSummaryGenerator._clean_value(
                    latest.get("diagnosis")
                ),

            "cancer_type":
                CaseSummaryGenerator._clean_value(
                    latest.get("cancer_type")
                ),

            "cancer_site":
                CaseSummaryGenerator._clean_value(
                    latest.get("cancer_site")
                ),

            "tumor_size":
                CaseSummaryGenerator._clean_value(
                    latest.get("tumor_size")
                ),

            "cancer_stage":
                CaseSummaryGenerator._clean_value(
                    latest.get("cancer_stage")
                ),

            "histopathology":
                CaseSummaryGenerator._clean_value(
                    latest.get("histopathology")
                ),

            # Clinical
            "clinical_history":
                CaseSummaryGenerator._clean_value(
                    latest.get("clinical_history")
                ),

            "current_status":
                CaseSummaryGenerator._clean_value(
                    latest.get("impression")
                ),

            # Treatment
            "treatments":
                CaseSummaryGenerator._clean_list(
                    latest.get("treatments")
                ),

            "medications":
                CaseSummaryGenerator._clean_list(
                    latest.get("medications")
                ),

            "follow_up":
                CaseSummaryGenerator._clean_value(
                    latest.get("follow_up")
                ),

            "recommendations":
                CaseSummaryGenerator._clean_list(
                    latest.get("recommendations")
                ),

            # Findings
            "key_findings": findings
        }

        return summary