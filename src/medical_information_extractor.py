from llm.llm_engine import generate_response
import re


# --------------------------------------------------
# Canonical field -> accepted label spellings
# --------------------------------------------------
# The model rarely reproduces our field names byte-for-byte - it drops
# words, swaps hyphens for spaces ("CA-125" -> "CA 125"), shortens
# "ER Status" to "ER", etc. Matching one fixed string per field broke on
# any of these. Each field now accepts a small set of known spellings.
FIELD_ALIASES = {
    "Patient Name": ["Patient Name", "Name"],
    "Age": ["Age"],
    "Gender": ["Gender", "Sex"],
    "Report Type": ["Report Type"],
    "Report Date": ["Report Date", "Date"],
    "Diagnosis": ["Diagnosis"],
    "Cancer Type": ["Cancer Type"],
    "Cancer Site": ["Cancer Site", "Site"],
    "Tumor Size": ["Tumor Size", "Tumour Size", "Size"],
    "Tumor Grade": ["Tumor Grade", "Tumour Grade", "Grade"],
    "Cancer Stage": ["Cancer Stage", "Stage", "TNM Stage"],
    "Histopathology": ["Histopathology", "Histology"],
    "Clinical History": ["Clinical History", "History"],
    "Findings": ["Findings", "Finding"],
    "Impression": ["Impression", "Impressions"],
    "Current Status": ["Current Status", "Status", "Disease Status"],
    "Treatment": ["Treatment", "Treatments", "Treatment Given"],
    "Medications": ["Medications", "Medication", "Drugs"],
    "Follow-up": ["Follow-up", "Follow up", "Followup"],
    "Recommendations": ["Recommendations", "Recommendation", "Advice"],
    "Abnormal Values": ["Abnormal Values"],
    "ER Status": ["ER Status", "ER", "Estrogen Receptor"],
    "PR Status": ["PR Status", "PR", "Progesterone Receptor"],
    "HER2 Status": ["HER2 Status", "HER2", "HER2/neu", "HER-2"],
    "Ki-67": ["Ki-67", "Ki67", "Ki 67"],
    "Hemoglobin": ["Hemoglobin", "Haemoglobin", "Hb"],
    "WBC": ["WBC", "White Blood Cells", "White Cell Count"],
    "Platelets": ["Platelets", "Platelet Count"],
    "Creatinine": ["Creatinine"],
    "AST": ["AST", "SGOT"],
    "ALT": ["ALT", "SGPT"],
    "ALP": ["ALP", "Alkaline Phosphatase"],
    "Bilirubin": ["Bilirubin"],
    "CEA": ["CEA"],
    "CA 15-3": ["CA 15-3", "CA15-3", "CA 15.3"],
    "CA-125": ["CA-125", "CA 125", "CA125"],
    "PSA": ["PSA"],
    "Key Medical Terms": ["Key Medical Terms", "Medical Terms", "Key Terms"],
}

NOT_MENTIONED_VALUES = {
    "",
    "not mentioned",
    "not mentioned.",
    "none mentioned",
    "none",
    "unknown",
    "not explicitly stated",
    "n/a",
    "na",
}

# Matches a line that is itself a "Label:" boundary, e.g. "Diagnosis:" or
# "Cancer Stage: T2 N0 M0". Deliberately generic - rather than one regex
# per known field - so we still recognise the boundary even when the
# model writes a header we don't otherwise care about (e.g. "Biomarkers:")
# or repeats a label. Either way, it stops the PREVIOUS field's value
# from bleeding into it, which is what the old fixed-list search couldn't
# guarantee.
_LABEL_LINE = re.compile(r"^[ \t]*([A-Za-z][A-Za-z0-9 /\-]{1,40}):[ \t]*(.*)$")


def _parse_labeled_blocks(text: str) -> dict:
    """
    Split free-form LLM output into {lowercase label: value} blocks.

    Any line that looks like "Label:" (optionally with trailing content on
    the same line) starts a new block; every following line up to the next
    such boundary belongs to that block. This makes parsing resilient to
    labels the model paraphrases, reorders, skips, or repeats - all of
    which silently corrupted the previous fixed-position substring search.
    """
    blocks = {}
    current_key = None
    current_lines = []

    def flush():
        if current_key is None:
            return
        value = "\n".join(current_lines).strip()
        # If a label appears more than once, keep whichever occurrence
        # is more informative rather than the first or the last blindly.
        if current_key not in blocks or len(value) > len(blocks[current_key]):
            blocks[current_key] = value

    for raw_line in text.split("\n"):
        match = _LABEL_LINE.match(raw_line.rstrip())
        if match:
            flush()
            current_key = match.group(1).strip().lower()
            trailing = match.group(2).strip()
            current_lines = [trailing] if trailing else []
        elif current_key is not None:
            current_lines.append(raw_line)

    flush()
    return _split_embedded_labels(blocks)


# A second, narrower field list used only to catch a label that lands
# mid-line rather than at the start of one - e.g. the model writing
# "Findings: Impression: Blood counts adequate for treatment." as a
# single line, which the line-start parser above can't split on its own.
# Deliberately limited to longer/distinctive labels (not bare "Grade",
# "Site", "Status" etc.) to avoid false-splitting on ordinary sentences.
_EMBEDDED_LABEL_ALIASES = {
    "Diagnosis": ["Diagnosis"],
    "Cancer Type": ["Cancer Type"],
    "Cancer Site": ["Cancer Site"],
    "Cancer Stage": ["Cancer Stage", "TNM Stage"],
    "Tumor Size": ["Tumor Size", "Tumour Size"],
    "Tumor Grade": ["Tumor Grade", "Tumour Grade"],
    "Histopathology": ["Histopathology", "Histology"],
    "Clinical History": ["Clinical History"],
    "Findings": ["Findings"],
    "Impression": ["Impression", "Impressions"],
    "Current Status": ["Current Status", "Disease Status"],
    "Treatment": ["Treatment", "Treatments", "Treatment Given"],
    "Medications": ["Medications", "Medication"],
    "Follow-up": ["Follow-up", "Follow up", "Followup"],
    "Recommendations": ["Recommendations", "Recommendation"],
    "Abnormal Values": ["Abnormal Values"],
    "Key Medical Terms": ["Key Medical Terms"],
}

_embedded_alias_list = sorted(
    {alias for aliases in _EMBEDDED_LABEL_ALIASES.values() for alias in aliases},
    key=len,
    reverse=True,
)
_EMBEDDED_LABEL_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _embedded_alias_list) + r")\s*:\s*",
    re.IGNORECASE,
)


def _split_embedded_labels(blocks: dict) -> dict:
    """
    If a known field label turns up embedded inside another field's
    captured value (two labels on one physical line), cut the value there
    and hand the remainder to the embedded field instead of letting it
    bleed into the field that happened to come first.
    """
    for key in list(blocks.keys()):
        value = blocks[key]
        match = _EMBEDDED_LABEL_RE.search(value)
        if not match:
            continue

        embedded_label = match.group(1).strip().lower()
        if embedded_label == key:
            continue

        before = value[:match.start()].strip()
        after = value[match.end():].strip()

        blocks[key] = before

        if after and (embedded_label not in blocks or len(after) > len(blocks[embedded_label])):
            blocks[embedded_label] = after

    return blocks


def _lookup(blocks: dict, canonical_field: str) -> str:
    """
    Look up a field's value by canonical name, trying every known
    alias spelling until one is found among the parsed blocks.
    """
    for alias in FIELD_ALIASES.get(canonical_field, [canonical_field]):
        value = blocks.get(alias.lower())
        if value is not None and value.strip():
            return value.strip()

    return "Not Mentioned"


def extract_medical_information(report_text: str, report_type: str):
    """
    Extract structured medical information from a medical report.
    Returns both flat fields and richer nested dictionaries for:
    - tumor_details
    - biomarkers
    - lab_results
    """

    prompt = f"""
You are an expert medical information extraction system.

Your ONLY job is to extract information explicitly written in the report.

IMPORTANT RULES

- Never guess.
- Never infer missing information.
- Never expand abbreviations.
- Never explain medical terminology.
- Never create diagnoses that are not explicitly written.
- If OCR makes a field unreadable, write "Unclear due to OCR".
- If information is absent, write exactly "Not Mentioned".
- Ignore hospital names, page headers, page footers, doctor names, registration numbers, logos and addresses.
- Preserve dates exactly as written.
- Keep the output strictly factual and concise.
- Do NOT include section dividers, separator lines, or template headings.
- Print every field label below exactly once. Never repeat a label, even if the same information could belong under two fields.
- "Treatment" means what has been or is being administered (procedures, chemotherapy, radiotherapy, medications already given or in progress).
- "Recommendations" means advice or suggested next steps written by the reporting clinician (for example follow-up tests, referrals, monitoring). If the report does not separately state any recommendation beyond the treatment already covered under "Treatment", write "Not Mentioned" for Recommendations instead of repeating the Treatment text.

Extract ONLY these fields.

Patient Name:
Age:
Gender:

Report Type:
Report Date:

Diagnosis:

Cancer Type:
(Examples: Breast Cancer, Lung Cancer, Colon Cancer, Leukemia, Lymphoma)

Cancer Site:

Tumor Size:
Tumor Grade:
Cancer Stage:

Histopathology:

Clinical History:

Findings:

Impression:

Current Status:

Treatment:

Medications:

Follow-up:

Recommendations:

Abnormal Values:

-------------------------
BIOMARKERS
-------------------------

ER Status:
PR Status:
HER2 Status:
Ki-67:

-------------------------
LAB RESULTS
-------------------------

Hemoglobin:
WBC:
Platelets:
Creatinine:
AST:
ALT:
ALP:
Bilirubin:
CEA:
CA 15-3:
CA-125:
PSA:

Key Medical Terms:

OUTPUT RULES

- Return plain text only.
- Do NOT use Markdown.
- Do NOT use **bold**.
- Do NOT use bullet points unless multiple values exist.
- If a field is absent return exactly:
Not Mentioned

Medical Report:

{report_text}
"""

    extracted_text = generate_response(prompt).strip()

    # ---------------------------------------
    # Cleanup LLM formatting
    # ---------------------------------------
    extracted_text = extracted_text.replace("\r\n", "\n").replace("\r", "\n")
    extracted_text = re.sub(r"\*\*", "", extracted_text)
    extracted_text = re.sub(r"__+", "", extracted_text)
    extracted_text = re.sub(r"`+", "", extracted_text)

    # Remove separator/template artefacts before parsing, so a stray
    # "-------------------------" or bare "BIOMARKERS" heading line can
    # never end up captured as (or inside) a field's value.
    extracted_text = re.sub(r"(?im)^\s*-{2,}\s*$", "", extracted_text)
    extracted_text = re.sub(
        r"(?im)^\s*(BIOMARKERS|LAB RESULTS|OTHER)\s*:?\s*-*\s*$",
        "",
        extracted_text,
    )
    extracted_text = re.sub(r"-{3,}", "", extracted_text)
    extracted_text = re.sub(r"\n{3,}", "\n\n", extracted_text)

    replacements = {
        "Not explicitly stated": "Not Mentioned",
        "None mentioned": "Not Mentioned",
        "None Mentioned": "Not Mentioned",
        "Unknown": "Not Mentioned",
        "not explicitly stated": "Not Mentioned",
        "none mentioned": "Not Mentioned",
        "none": "Not Mentioned",
        "unknown": "Not Mentioned",
    }

    for old, new in replacements.items():
        extracted_text = extracted_text.replace(old, new)

    extracted_text = extracted_text.strip()

    # ---------------------------------------
    # Parse once into labeled blocks, then look fields up by canonical
    # name (with aliases) instead of re-scanning the raw text per field.
    # ---------------------------------------
    blocks = _parse_labeled_blocks(extracted_text)

    def get_field(canonical_name: str) -> str:
        value = _lookup(blocks, canonical_name)
        value = re.sub(r"^[\-\*\•\s]+", "", value).strip()

        if value.lower() in NOT_MENTIONED_VALUES:
            return "Not Mentioned"

        return value

    # ---------------------------------------
    # Helper for list fields
    # ---------------------------------------
    def clean_list(value: str):
        if value == "Not Mentioned":
            return []

        items = []
        seen = set()

        for line in value.split("\n"):
            line = line.strip("-•* ").strip()

            if not line:
                continue

            if line.lower() in NOT_MENTIONED_VALUES:
                continue

            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            items.append(line)

        return items

    # ---------------------------------------
    # Helper for scalar fields
    # ---------------------------------------
    def clean_value(value: str) -> str:
        if not value:
            return "Not Mentioned"

        value = value.strip()
        value = re.sub(r"^[\-\*\•\s]+", "", value).strip()
        value = re.sub(r"\s{2,}", " ", value)

        if value.lower() in NOT_MENTIONED_VALUES:
            return "Not Mentioned"

        return value

    # ---------------------------------------
    # Extract core fields
    # ---------------------------------------
    medical_terms = clean_list(get_field("Key Medical Terms"))
    treatments = clean_list(get_field("Treatment"))
    medications = clean_list(get_field("Medications"))
    recommendations = clean_list(get_field("Recommendations"))

    # A recommendation that just restates a treatment verbatim isn't
    # adding information - keep it only if it says something different.
    treatment_keys = {t.strip().lower() for t in treatments}
    recommendations = [r for r in recommendations if r.strip().lower() not in treatment_keys]

    # ---------------------------------------
    # Nested structured outputs
    # ---------------------------------------
    tumor_details = {
        "size": clean_value(get_field("Tumor Size")),
        "grade": clean_value(get_field("Tumor Grade")),
        "stage": clean_value(get_field("Cancer Stage")),
        "histopathology": clean_value(get_field("Histopathology")),
    }

    biomarkers = {
        "ER": clean_value(get_field("ER Status")),
        "PR": clean_value(get_field("PR Status")),
        "HER2": clean_value(get_field("HER2 Status")),
        "Ki-67": clean_value(get_field("Ki-67")),
    }

    lab_results = {
        "Hemoglobin": clean_value(get_field("Hemoglobin")),
        "WBC": clean_value(get_field("WBC")),
        "Platelets": clean_value(get_field("Platelets")),
        "Creatinine": clean_value(get_field("Creatinine")),
        "AST": clean_value(get_field("AST")),
        "ALT": clean_value(get_field("ALT")),
        "ALP": clean_value(get_field("ALP")),
        "Bilirubin": clean_value(get_field("Bilirubin")),
        "CEA": clean_value(get_field("CEA")),
        "CA 15-3": clean_value(get_field("CA 15-3")),
        "CA-125": clean_value(get_field("CA-125")),
        "PSA": clean_value(get_field("PSA")),
    }

    # ---------------------------------------
    # Return structured information
    # ---------------------------------------
    return {
        # Patient
        "patient_name": get_field("Patient Name"),
        "age": get_field("Age"),
        "gender": get_field("Gender"),

        # Report
        "report_type": report_type,
        "report_date": get_field("Report Date"),

        # Diagnosis
        "diagnosis": get_field("Diagnosis"),
        "cancer_type": get_field("Cancer Type"),
        "cancer_site": get_field("Cancer Site"),

        # Flat tumour fields (kept for compatibility)
        "tumor_size": get_field("Tumor Size"),
        "cancer_stage": get_field("Cancer Stage"),
        "histopathology": get_field("Histopathology"),
        "tumor_grade": get_field("Tumor Grade"),

        # Nested tumour structure
        "tumor_details": tumor_details,

        # Clinical
        "clinical_history": get_field("Clinical History"),
        "findings": get_field("Findings"),
        "impression": get_field("Impression"),
        "current_status": get_field("Current Status"),

        # Treatment
        "treatments": treatments,
        "medications": medications,
        "follow_up": get_field("Follow-up"),

        # Other
        "recommendations": recommendations,
        "abnormal_values": get_field("Abnormal Values"),
        "medical_terms": medical_terms,

        # Nested structures
        "biomarkers": biomarkers,
        "lab_results": lab_results,

        # Raw LLM output
        "structured_summary": extracted_text
    }