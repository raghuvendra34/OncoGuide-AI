"""
=========================================================
OncoGuide AI v2
Next Generation Oncology Intelligence Platform

Author: Raghuvendra Kumar
Architecture:
    • Modern Dashboard
    • Multi PDF Pipeline
    • RAG
    • Medical Timeline
    • Cross Report Reasoning
    • Chat Assistant
    • Evidence Retrieval
=========================================================
"""

from __future__ import annotations

import html
import os
import re
import json
import tempfile
import traceback
from pathlib import Path
from typing import Dict
from typing import List
from typing import Any
from typing import Optional

import streamlit as st
import streamlit.components.v1 as components


# ============================================================
# INTERNAL MODULES
# ============================================================

from src.report_classifier import detect_report_type
from src.memory import ConversationMemory

from src.pdf_reader import extract_text_from_pdf
from src.text_chunker import chunk_text

from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store

from src.medical_information_extractor import (
    extract_medical_information,
)

from src.cross_report_reasoner import (
    CrossReportReasoner,
)

from src.case_summary_generator import (
    CaseSummaryGenerator,
)

from src.patient_timeline import (
    PatientTimeline,
)

from src.patient_journey import (
    PatientJourneyGenerator,
)

from src.medical_terms import (
    simplify_terms,
)

from src.report_explainer import (
    explain_report,
    generate_patient_summary,
)

from src.retrieval_service import (
    retrieve_context,
)

from src.evidence_analyzer import (
    analyze_evidence,
)

from services.chat_service import (
    answer_question,
)

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="OncoGuide AI v2",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONSTANTS
# ============================================================

APP_NAME = "OncoGuide AI"
APP_VERSION = "v2"

PRIMARY = "#27C7A8"
ACCENT = "#4BE1C1"

BACKGROUND = "#08110F"
CARD = "#111D19"
CARD_ALT = "#16231E"

TEXT = "#F4F5F3"
TEXT_DIM = "#8D9D98"

BORDER = "#233A33"

MAX_UPLOAD = 20

# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_SESSION = {

    "reports": [],

    "vector_db": None,

    "messages": [],

    "memory": ConversationMemory(),

    "patient_summary": {},

    "case_summary": {},

    "patient_journey": "",

    "patient_timeline": PatientTimeline(),

    "pipeline_complete": False,

    "processing": False,

}

for key, value in DEFAULT_SESSION.items():

    if key not in st.session_state:

        st.session_state[key] = value

# ============================================================
# GLOBAL HELPERS
# ============================================================

def clean_text(value: Any) -> str:

    if value is None:
        return ""

    if isinstance(value, (int, float)):
        return str(value)

    text = str(value)

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = text.strip()

    if not text:
        return ""

    lines = []

    seen = set()

    for line in text.split("\n"):

        line = line.strip()

        if not line:
            continue

        key = line.lower()

        if key in seen:
            continue

        seen.add(key)

        lines.append(line)

    return "\n".join(lines)


def html_escape(value: Any) -> str:

    value = clean_text(value)

    if value == "":
        return "Not Available"

    return html.escape(value)


def has_content(value: Any) -> bool:

    if value is None:
        return False

    if isinstance(value, dict):

        return any(has_content(v) for v in value.values())

    if isinstance(value, list):

        return any(has_content(v) for v in value)

    return clean_text(value) != ""


def card(title: str, body: str):

    st.markdown(
        f"""
<div class="card">

<h4>{html_escape(title)}</h4>

{body}

</div>
""",
        unsafe_allow_html=True,
    )

# ============================================================
# DESIGN SYSTEM
# ============================================================

GLOBAL_CSS = f"""

<style>

:root{{
--bg:{BACKGROUND};
--card:{CARD};
--card2:{CARD_ALT};
--primary:{PRIMARY};
--accent:{ACCENT};
--text:{TEXT};
--muted:{TEXT_DIM};
--border:{BORDER};
}}

html,
body,
.stApp{{

background:var(--bg);

color:var(--text);

}}

[data-testid="stSidebar"]{{
background:#0D1613;
}}

.block-container{{
padding-top:2rem;
max-width:1450px;
}}

h1,h2,h3,h4{{
color:white;
}}

.card{{
background:var(--card);

border:1px solid var(--border);

border-radius:18px;

padding:24px;

margin-bottom:18px;

box-shadow:
0 0 0 rgba(0,0,0,0);

transition:.25s;
}}

.card:hover{{

transform:translateY(-3px);

border-color:var(--primary);

}}

.metric-card{{

background:linear-gradient(
180deg,
#15231F,
#101916
);

border-radius:18px;

padding:22px;

text-align:center;

border:1px solid var(--border);

}}

.metric-number{{

font-size:34px;

font-weight:700;

color:var(--accent);

}}

.metric-label{{

color:var(--muted);

font-size:14px;

}}

.hero{{

padding:48px;

border-radius:24px;

background:

linear-gradient(
135deg,
#143028,
#08110F
);

border:1px solid var(--border);

margin-bottom:28px;

}}

.hero-title{{

font-size:48px;

font-weight:800;

}}

.hero-sub{{

font-size:18px;

color:var(--muted);

margin-top:8px;

}}

section{{

margin-bottom:40px;

}}

</style>

"""

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ============================================================
# HERO
# ============================================================

st.markdown(
f"""
<div class="hero">

<div class="hero-title">

🎗️ {APP_NAME}

</div>

<div class="hero-sub">

AI-powered Oncology Decision Support Platform

Built using Retrieval-Augmented Generation,
Medical Knowledge Graphs,
Patient Timelines,
Evidence Retrieval,
and Cross-Report Clinical Reasoning.

</div>

</div>
""",
unsafe_allow_html=True,
)

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Navigation")

    st.divider()

    # st.markdown("### Pipeline")

    # st.caption("Upload → Extract → RAG → Timeline → Chat")

    # st.divider()

    uploaded_files = st.file_uploader(

        "Medical Reports",

        type=["pdf"],

        accept_multiple_files=True,

    )

    st.divider()

    st.markdown("### Status")

    st.write(
        "Reports:",
        len(st.session_state.reports),
    )

    st.write(
        "Vector DB:",
        " Ready"
        if st.session_state.vector_db
        else "Not Built",
    )

    st.write(
        "Timeline:",
        len(
            st.session_state.patient_timeline.get_timeline()
        ),
    )

# ============================================================
# DASHBOARD METRICS
# ============================================================

c1,c2,c3,c4=st.columns(4)

with c1:

    st.markdown(
"""
<div class="metric-card">

<div class="metric-number">
0
</div>

<div class="metric-label">
Reports
</div>

</div>
""",
unsafe_allow_html=True,
)

with c2:

    st.markdown(
"""
<div class="metric-card">

<div class="metric-number">
0
</div>

<div class="metric-label">
Timeline Events
</div>

</div>
""",
unsafe_allow_html=True,
)

with c3:

    st.markdown(
"""
<div class="metric-card">

<div class="metric-number">
0
</div>

<div class="metric-label">
Clinical Findings
</div>

</div>
""",
unsafe_allow_html=True,
)

with c4:

    st.markdown(
"""
<div class="metric-card">

<div class="metric-number">
Ready
</div>

<div class="metric-label">
AI Engine
</div>

</div>
""",
unsafe_allow_html=True,
)

# ============================================================
# REPORT PROCESSING STATE
# ============================================================

new_reports = []
all_reports = []
complete_text = ""
chunks = []
embedding_model = None

if uploaded_files:

    existing_files = {
        report["filename"]
        for report in st.session_state.reports
    }

    new_reports = []

    with st.spinner("Processing medical reports..."):

        st.session_state.processing = True

        for uploaded_file in uploaded_files:

            if uploaded_file.name in existing_files:
                continue

            # ------------------------------------------
            # Save uploaded PDF
            # ------------------------------------------

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temp_file:

                temp_file.write(uploaded_file.read())

                pdf_path = temp_file.name

            # ------------------------------------------
            # Extract Text
            # ------------------------------------------

            report_text = extract_text_from_pdf(
                pdf_path
            )

            if not clean_text(report_text):

                st.warning(
                    f"Unable to read {uploaded_file.name}"
                )

                continue

            # ------------------------------------------
            # Detect Report Type
            # ------------------------------------------

            report_type = detect_report_type(
                report_text
            )

            # ------------------------------------------
            # Extract Structured Information
            # ------------------------------------------

            structured_info = (
                extract_medical_information(
                    report_text,
                    report_type,
                )
            )

            structured_info["report_type"] = report_type

            # ------------------------------------------
            # Store Report
            # ------------------------------------------

            report = {

                "filename": uploaded_file.name,

                "type": report_type,

                "text": report_text,

                "structured_info": structured_info,

            }

            st.session_state.reports.append(
                report
            )

            new_reports.append(
                structured_info
            )

            # ------------------------------------------
            # Timeline
            # ------------------------------------------

            st.session_state.patient_timeline.add_report(

                report_text=report_text,

                report_type=report_type,

                structured_info=structured_info,

            )

        st.session_state.processing = False

# ============================================================
# BUILD CROSS REPORT SUMMARY
# ============================================================

if new_reports:

    all_reports = [

        report["structured_info"]

        for report in st.session_state.reports

    ]

    try:

        reasoner = CrossReportReasoner(
            all_reports
        )

        st.session_state.patient_summary = (

            reasoner.build_summary()

        )

    except Exception:

        traceback.print_exc()

# ============================================================
# CASE SUMMARY
# ============================================================

if new_reports:

    try:

        st.session_state.case_summary = (

            CaseSummaryGenerator.generate(
                all_reports
            )

        )

    except Exception:

        traceback.print_exc()

# ============================================================
# PATIENT JOURNEY
# ============================================================

if new_reports:

    try:

        reports_for_journey = []

        for report in st.session_state.reports:

            reports_for_journey.append(

                {

                    "filename":
                    report["filename"],

                    "report_type":
                    report["type"],

                    "text":
                    report["text"],

                }

            )

        journey_generator = (

            PatientJourneyGenerator()

        )

        st.session_state.patient_journey = (

            journey_generator.generate_journey(
                reports_for_journey
            )

        )

    except Exception:

        traceback.print_exc()

# ============================================================
# CREATE MASTER TEXT FOR RAG
# ============================================================

if new_reports:

    master_document = []

    for report in st.session_state.reports:

        structured_summary = (

            report["structured_info"].get(
                "structured_summary",
                "",
            )

        )

        document = f"""

==================================================

FILE

{report['filename']}

--------------------------------------------------

REPORT TYPE

{report['type']}

--------------------------------------------------

STRUCTURED SUMMARY

{structured_summary}

--------------------------------------------------

FULL REPORT

{report['text']}

==================================================

"""

        master_document.append(
            document
        )

    complete_text = "\n".join(
        master_document
    )

# ============================================================
# TEXT CHUNKING
# ============================================================

if new_reports:

    chunks = chunk_text(
        complete_text
    )

    st.session_state.total_chunks = len(
        chunks
    )

# ============================================================
# EMBEDDING MODEL
# ============================================================

if new_reports:

    embedding_model = get_embedding_model()

# ============================================================
# VECTOR DATABASE
# ============================================================

if new_reports:

    st.session_state.vector_db = (

        create_vector_store(

            chunks,

            embedding_model,

        )

    )

# ============================================================
# PIPELINE STATUS
# ============================================================

if new_reports:

    st.session_state.pipeline_complete = True

    st.success(
        f"""
Successfully processed
{len(new_reports)} report(s).

✓ Text Extraction

✓ Medical Information Extraction

✓ Timeline Generation

✓ Cross Report Reasoning

✓ Patient Summary

✓ Vector Database

✓ RAG Pipeline Ready
"""
    )

# ============================================================
# DASHBOARD DATA
# ============================================================

if st.session_state.pipeline_complete:

    reports = st.session_state.reports

    patient_summary = st.session_state.patient_summary

    case_summary = st.session_state.case_summary

    timeline = (
        st.session_state
        .patient_timeline
        .get_timeline()
    )

    findings = (
        case_summary.get(
            "key_findings",
            [],
        )
        if case_summary
        else []
    )

# ============================================================
# BLOCK 3
# MODERN MEDICAL DASHBOARD
# ============================================================

if st.session_state.pipeline_complete:

    # --------------------------------------------------------
    # LIVE METRICS
    # --------------------------------------------------------

    report_count = len(reports)

    timeline_events = len(timeline)

    finding_count = len(findings)

    cancer_type = (
        patient_summary.get("cancer_type")
        or "Unknown"
    )

    stage = (
        patient_summary.get("cancer_stage")
        or "Unknown"
    )

    status = (
        patient_summary.get("current_status")
        or "Unknown"
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-number">
{report_count}
</div>

<div class="metric-label">
Uploaded Reports
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    with c2:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-number">
{timeline_events}
</div>

<div class="metric-label">
Timeline Events
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    with c3:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-number">
{finding_count}
</div>

<div class="metric-label">
Key Findings
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    with c4:

        st.markdown(
            f"""
<div class="metric-card">

<div class="metric-number">
Ready
</div>

<div class="metric-label">
Vector Database
</div>

</div>
""",
            unsafe_allow_html=True,
        )

    st.divider()

# ============================================================
# REPORT LIBRARY
# ============================================================

    st.header("🎗️ Medical Report Library")

    cols = st.columns(3)

    for index, report in enumerate(reports):

        with cols[index % 3]:

            st.markdown(
                f"""
<div class="card">

<h4>{html_escape(report['filename'])}</h4>

<b>Type</b><br>

{html_escape(report['type'])}

<br><br>

<b>Characters</b><br>

{len(report['text'])}

</div>
""",
                unsafe_allow_html=True,
            )

# ============================================================
# PATIENT OVERVIEW
# ============================================================

    st.header("🎗️ Patient Overview")

    left, right = st.columns([1, 1])

    with left:

        overview = [

            (
                "Diagnosis",

                patient_summary.get(
                    "diagnosis"
                ),
            ),

            (
                "Cancer Type",

                patient_summary.get(
                    "cancer_type"
                ),
            ),

            (
                "Cancer Site",

                patient_summary.get(
                    "cancer_site"
                ),
            ),

            (
                "Stage",

                patient_summary.get(
                    "cancer_stage"
                ),
            ),

            (
                "Current Status",

                patient_summary.get(
                    "current_status"
                ),
            ),

        ]

        html_output = ""

        for title, value in overview:

            html_output += f"""

<p>

<b>{title}</b><br>

{html_escape(value)}

</p>

"""

        card(
            "Clinical Summary",
            html_output,
        )

    with right:

        patient = case_summary.get(
            "patient_information",
            {},
        )

        demographics = f"""

<p>

<b>Name</b><br>

{html_escape(patient.get("name"))}

</p>

<p>

<b>Age</b><br>

{html_escape(patient.get("age"))}

</p>

<p>

<b>Gender</b><br>

{html_escape(patient.get("gender"))}

</p>

"""

        card(
            "Patient Information",
            demographics,
        )

# ============================================================
# KEY FINDINGS
# ============================================================

    st.header("🎗️ Key Clinical Findings")

    if findings:

        for finding in findings:

            st.info(finding)

    else:

        st.info(
            "No findings detected."
        )

# ============================================================
# TREATMENTS
# ============================================================

    treatments = patient_summary.get(
        "treatments",
        [],
    )

    medications = patient_summary.get(
        "medications",
        [],
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🎗️ Treatments")

        if treatments:

            for treatment in treatments:

                st.success(
                    treatment
                )

        else:

            st.caption(
                "Not Available"
            )

    with col2:

        st.subheader("🎗️ Medications")

        if medications:

            for medicine in medications:

                st.success(
                    medicine
                )

        else:

            st.caption(
                "Not Available"
            )

# ============================================================
# BIOMARKERS
# ============================================================

    biomarkers = case_summary.get(
        "biomarkers",
        {},
    )

    if biomarkers:

        st.header("🎗️ Biomarkers")

        biomarker_table = []

        for key, value in biomarkers.items():

            biomarker_table.append(

                {

                    "Biomarker": key,

                    "Value": value,

                }

            )

        st.table(
            biomarker_table
        )

# ============================================================
# TUMOR DETAILS
# ============================================================

    tumor = case_summary.get(
        "tumor_details",
        {},
    )

    if tumor:

        st.header("🎗️ Tumor Profile")

        tumor_table = []

        for key, value in tumor.items():

            tumor_table.append(

                {

                    "Parameter": key,

                    "Value": value,

                }

            )

        st.table(
            tumor_table
        )

# ============================================================
# LAB RESULTS
# ============================================================

    labs = case_summary.get(
        "lab_results",
        {},
    )

    if labs:

        st.header("🎗️ Laboratory Results")

        lab_table = []

        for key, value in labs.items():

            lab_table.append(

                {

                    "Test": key,

                    "Result": value,

                }

            )

        st.table(
            lab_table
        )

# ============================================================
# PATIENT TIMELINE
# ============================================================

    st.header("🎗️ Patient Timeline")

    if timeline:

        for event in timeline:

            with st.container(border=True):

                left, right = st.columns(
                    [1, 4]
                )

                with left:

                    st.caption(
                        event.get(
                            "date",
                            "Unknown",
                        )
                    )

                with right:

                    st.markdown(
                        f"""
### {event.get("event","")}

**Report Type**

{event.get("report_type","")}
"""
                    )

    else:

        st.info(
            "Timeline unavailable."
        )
# ============================================================
# BLOCK 4
# PATIENT JOURNEY
# STRUCTURED REPORT VIEWER
# GLOSSARY
# AI EXPLANATION
# ============================================================

# ------------------------------------------------------------
# PATIENT JOURNEY
# ------------------------------------------------------------

if st.session_state.pipeline_complete:

    if st.session_state.patient_journey:

        st.header("🎗️ Patient Journey")

        card(

            "Clinical Journey",

            st.session_state.patient_journey,

        )

# ============================================================
# STRUCTURED REPORT VIEWER
# ============================================================

if st.session_state.pipeline_complete:

    st.header("🎗️ Structured Medical Reports")

    for report in reports:

        info = report["structured_info"]

        with st.expander(
            report["filename"],
            expanded=False,
        ):

            left, right = st.columns(2)

            # ------------------------------------------
            # LEFT COLUMN
            # ------------------------------------------

            with left:

                st.subheader("Diagnosis")

                st.write(

                    info.get(
                        "diagnosis",
                        "Not Available",
                    )

                )

                st.subheader("Findings")

                st.write(

                    info.get(
                        "findings",
                        "Not Available",
                    )

                )

                st.subheader("Impression")

                st.write(

                    info.get(
                        "impression",
                        "Not Available",
                    )

                )

            # ------------------------------------------
            # RIGHT COLUMN
            # ------------------------------------------

            with right:

                st.subheader("Treatment")

                treatments = info.get(
                    "treatments",
                    [],
                )

                if isinstance(
                    treatments,
                    list,
                ):

                    if treatments:

                        for item in treatments:

                            st.success(item)

                    else:

                        st.caption(
                            "No treatment extracted."
                        )

                else:

                    st.write(
                        treatments
                    )

                st.subheader(
                    "Recommendations"
                )

                recommendations = info.get(
                    "recommendations",
                    [],
                )

                if isinstance(
                    recommendations,
                    list,
                ):

                    if recommendations:

                        for item in recommendations:

                            st.info(item)

                    else:

                        st.caption(
                            "No recommendations."
                        )

                else:

                    st.write(
                        recommendations
                    )

            st.divider()

            st.subheader(
                "Structured Summary"
            )

            st.write(

                info.get(
                    "structured_summary",
                    "Unavailable",
                )

            )

# ============================================================
# ORIGINAL REPORT TEXT
# ============================================================

if st.session_state.pipeline_complete:

    st.header("🎗️ Original Reports")

    for report in reports:

        with st.expander(

            f"Raw Report • {report['filename']}",

            expanded=False,

        ):

            st.text_area(

                "Text",

                value=report["text"],

                height=350,

                disabled=True,

                key=f"raw_{report['filename']}",

            )

# ============================================================
# MEDICAL GLOSSARY
# ============================================================

if st.session_state.pipeline_complete:

    st.header("🎗️ Plain Language Glossary")

    glossary_placeholder = st.empty()

    if st.button(

        "Generate Glossary",

        use_container_width=True,

    ):

        with st.spinner(

            "Simplifying medical terminology..."

        ):

            combined_text = "\n".join(

                report["text"]

                for report in reports

            )

            glossary = simplify_terms(

                combined_text

            )

        glossary_placeholder.markdown(

            glossary

        )

# ============================================================
# AI REPORT EXPLANATION
# ============================================================

if st.session_state.pipeline_complete:

    st.header("🎗️ AI Report Analysis")

    col1, col2 = st.columns(2)

    explain_clicked = col1.button(

        "Explain Reports",

        use_container_width=True,

    )

    summary_clicked = col2.button(

        "Generate Patient Summary",

        use_container_width=True,

    )

# ------------------------------------------------------------
# FULL REPORT EXPLANATION
# ------------------------------------------------------------

    if explain_clicked:

        combined_text = "\n\n".join(

            report["text"]

            for report in reports

        )

        with st.spinner(

            "Generating explanation..."

        ):

            try:

                explanation = explain_report(

                    combined_text

                )

                card(

                    "AI Explanation",

                    explanation,

                )

            except Exception:

                st.error(

                    traceback.format_exc()

                )

# ------------------------------------------------------------
# AI SUMMARY
# ------------------------------------------------------------

    if summary_clicked:

        combined_text = "\n\n".join(

            report["text"]

            for report in reports

        )

        with st.spinner(

            "Building summary..."

        ):

            try:

                summary = (

                    generate_patient_summary(

                        combined_text

                    )

                )

                card(

                    "Patient Summary",

                    summary,

                )

            except Exception:

                st.error(

                    traceback.format_exc()

                )

# ============================================================
# RAG STATUS PANEL
# ============================================================

if st.session_state.pipeline_complete:

    st.header("🎗️ Retrieval Database")

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(

            "Documents",

            len(reports),

        )

    with col2:

        st.metric(

            "Chunks",

            st.session_state.get(

                "total_chunks",

                0,

            ),

        )

    with col3:

        st.metric(

            "Embedding Store",

            "Ready"

            if st.session_state.vector_db

            else "Unavailable",

        )
# ============================================================
# BLOCK 5
# ONCOGUIDE AI CHAT (RAG)
# ============================================================

st.header("🎗️ OncoGuide AI Assistant")

st.caption(
    "Ask questions about diagnosis, treatment, reports, biomarkers, staging, prognosis, medications, timeline, or any uploaded document."
)

# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        if message["role"] == "assistant":

            st.markdown(
                "**🩺 OncoGuide AI**"
            )

        st.markdown(
            message["content"]
        )

# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask something about your reports..."
)

# ============================================================
# USER MESSAGE
# ============================================================

if question:

    st.session_state.messages.append(

        {

            "role": "user",

            "content": question,

        }

    )

    st.session_state.memory.add_message(

        "user",

        question,

    )

    with st.chat_message("user"):

        st.markdown(question)

# ============================================================
# ASSISTANT RESPONSE
# ============================================================

    with st.chat_message("assistant"):

        st.markdown(
            "**🎗️ OncoGuide AI**"
        )

        # ----------------------------------------------------
        # CHECK VECTOR DATABASE
        # ----------------------------------------------------

        if st.session_state.vector_db is None:

            answer = (
                "Please upload one or more medical reports first."
            )

            st.warning(answer)

            retrieved_documents = []

            evidence_analysis = None

        else:

            try:

                # --------------------------------------------
                # Retrieve Context
                # --------------------------------------------

                context = retrieve_context(

                    st.session_state.vector_db,

                    question,

                )

                # --------------------------------------------
                # Generate Answer
                # --------------------------------------------

                answer, detected_report = (

                    answer_question(

                        question=question,

                        context=context,

                        memory=st.session_state.memory,

                        patient_summary=st.session_state.patient_summary,

                    )

                )

                # --------------------------------------------
                # Similarity Search
                # --------------------------------------------

                retrieved_documents = (

                    st.session_state.vector_db

                    .similarity_search(

                        question,

                        k=5,

                    )

                )

                # --------------------------------------------
                # Evidence Analysis
                # --------------------------------------------

                retrieved_chunks = [

                    document.page_content

                    for document in retrieved_documents

                ]

                evidence_analysis = (

                    analyze_evidence(

                        question=question,

                        retrieved_chunks=retrieved_chunks,

                    )

                )

            except Exception:

                answer = traceback.format_exc()

                retrieved_documents = []

                evidence_analysis = None

        # ----------------------------------------------------
        # MAIN ANSWER
        # ----------------------------------------------------

        st.markdown(answer)

        # ----------------------------------------------------
        # CONFIDENCE PANEL
        # ----------------------------------------------------

        if evidence_analysis:

            confidence = evidence_analysis.get(

                "confidence",

                "Unknown",

            )

            quality = evidence_analysis.get(

                "quality",

                "Unknown",

            )

            reason = evidence_analysis.get(

                "reason",

                "",

            )

            st.info(

                f"""

### Evidence Quality

**Confidence**

{confidence}

**Quality**

{quality}

**Reason**

{reason}

"""

            )

        # ----------------------------------------------------
        # DETECTED REPORT
        # ----------------------------------------------------

        if detected_report:

            st.success(

                f"Relevant Report: {detected_report}"

            )

        # ----------------------------------------------------
        # SOURCE DOCUMENTS
        # ----------------------------------------------------

        if retrieved_documents:

            with st.expander(

                "Retrieved Evidence",

                expanded=False,

            ):

                for index, document in enumerate(

                    retrieved_documents,

                    start=1,

                ):

                    st.markdown(

                        f"### Evidence {index}"

                    )

                    st.write(

                        document.page_content

                    )

                    st.divider()

# ============================================================
# SAVE CHAT
# ============================================================

    st.session_state.messages.append(

        {

            "role": "assistant",

            "content": answer,

            "sources": [

                document.page_content

                for document in retrieved_documents

            ],

        }

    )

    st.session_state.memory.add_message(

        "assistant",

        answer,

    )

# ============================================================
# MEMORY SUMMARIZATION
# ============================================================

    try:

        st.session_state.memory.summarize()

    except Exception:

        pass

# ============================================================
# CONVERSATION CONTROLS
# ============================================================

st.divider()

left, right = st.columns(2)

with left:

    if st.button(

        "Clear Conversation",

        use_container_width=True,

    ):

        st.session_state.messages = []

        st.session_state.memory = ConversationMemory()

        st.rerun()

with right:

    if st.button(

        "Clear Session",

        use_container_width=True,

    ):

        st.session_state.reports = []

        st.session_state.vector_db = None
        st.session_state.messages = []
        st.session_state.patient_summary = {}
        st.session_state.case_summary = {}
        st.session_state.patient_journey = ""
        st.session_state.patient_timeline = PatientTimeline()
        st.session_state.pipeline_complete = False
        st.session_state.total_chunks = 0
        st.session_state.memory = ConversationMemory()

        st.rerun()

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    f"""
{APP_NAME} {APP_VERSION}

AI-assisted oncology report interpretation using
Retrieval-Augmented Generation (RAG),
cross-report clinical reasoning,
patient timeline generation,
vector search,
and evidence-grounded responses.

This application is intended for educational and decision-support purposes only.
It is not a substitute for professional medical advice.
"""
)