import tempfile
import traceback

import streamlit as st

from src.pdf_reader import extract_text_from_pdf
from src.report_classifier import detect_report_type
from src.text_chunker import chunk_text
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store

from src.medical_terms import simplify_terms
from src.medical_information_extractor import (
    extract_medical_information,
)

from src.cross_report_reasoner import CrossReportReasoner
from src.patient_timeline import PatientTimeline
from src.patient_journey import PatientJourneyGenerator
from src.case_summary_generator import CaseSummaryGenerator

from src.memory import ConversationMemory
from src.retrieval_service import retrieve_context
from src.chat_service import answer_question
from src.evidence_analyzer import analyze_evidence

from src.report_explainer import (
    explain_report,
    generate_patient_summary,
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="OncoGuide AI",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 OncoGuide AI")
st.caption("Understand Your Cancer Report with AI")

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

def initialize_session():

    defaults = {
        "reports": [],
        "vector_db": None,
        "embedding_model": get_embedding_model(),
        "memory": ConversationMemory(),
        "messages": [],
        "patient_timeline": PatientTimeline(),
        "patient_summary": None,
        "patient_journey": None,
        "case_summary": None,
        "report_explanation": None,
        "medical_terms": "",
        "patient_summary_text": "",
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialize_session()

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload Medical Reports",
    type=["pdf"],
    accept_multiple_files=True,
)

# ---------------------------------------------------
# REPORT PROCESSING
# ---------------------------------------------------

def process_reports(files):

    if not files:
        return

    new_reports = []

    for uploaded_file in files:

        if any(
            r["filename"] == uploaded_file.name
            for r in st.session_state.reports
        ):
            continue

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.read())
            pdf_path = tmp.name

        report_text = extract_text_from_pdf(pdf_path)

        report_type = detect_report_type(report_text)

        structured_info = extract_medical_information(
            report_text,
            report_type,
        )

        report_data = {
            "filename": uploaded_file.name,
            "text": report_text,
            "type": report_type,
            "structured_info": structured_info,
        }

        st.session_state.reports.append(report_data)
        new_reports.append(structured_info)

        st.session_state.patient_timeline.add_report(
            report_text=report_text,
            report_type=report_type,
        )

    if not new_reports:
        return

    # -----------------------------------------
    # VECTOR DATABASE
    # -----------------------------------------

    all_chunks = []

    for report in st.session_state.reports:

        all_chunks.extend(
            chunk_text(report["text"])
        )

    st.session_state.vector_db = create_vector_store(
        all_chunks,
        st.session_state.embedding_model,
    )

    # -----------------------------------------
    # CROSS REPORT SUMMARY
    # -----------------------------------------

    structured_reports = [
        r["structured_info"]
        for r in st.session_state.reports
    ]

    reasoner = CrossReportReasoner(
        structured_reports
    )

    st.session_state.patient_summary = (
        reasoner.build_summary()
    )

    # -----------------------------------------
    # CASE SUMMARY
    # -----------------------------------------

    st.session_state.case_summary = (
        CaseSummaryGenerator.generate(
            structured_reports
        )
    )

    # -----------------------------------------
    # PATIENT JOURNEY
    # -----------------------------------------

    try:

        journey = PatientJourneyGenerator()

        st.session_state.patient_journey = (
            journey.generate_journey(
                [
                    {
                        "filename": r["filename"],
                        "report_type": r["type"],
                        "text": r["text"],
                    }
                    for r in st.session_state.reports
                ]
            )
        )

    except Exception:

        st.session_state.patient_journey = (
            "Unable to generate patient journey."
        )

    # ---------------------------------------------------
# PROCESS REPORTS
# ---------------------------------------------------

if uploaded_files:
    process_reports(uploaded_files)

# ---------------------------------------------------
# UPLOADED REPORTS
# ---------------------------------------------------

if st.session_state.reports:

    st.subheader("📚 Uploaded Reports")

    for report in st.session_state.reports:

        st.write(
            f"📄 {report['filename']} - {report['type']}"
        )

# ---------------------------------------------------
# PATIENT TIMELINE
# ---------------------------------------------------

timeline = st.session_state.patient_timeline.get_timeline()

if timeline:

    st.subheader("🕒 Patient Medical Timeline")

    for event in timeline:

        with st.container(border=True):

            st.markdown(f"### 📅 {event['date']}")

            st.write("**Event:**")
            st.write(event["event"])

            st.write("**Report Type:**")
            st.write(event["report_type"])

# ---------------------------------------------------
# PATIENT JOURNEY
# ---------------------------------------------------

if st.session_state.patient_journey:

    st.subheader("🩺 AI Patient Journey")

    with st.container(border=True):

        st.markdown(
            st.session_state.patient_journey
        )

# ---------------------------------------------------
# CROSS REPORT SUMMARY
# ---------------------------------------------------

summary = st.session_state.patient_summary

if summary:

    st.subheader("🧠 Cross Report Patient Summary")

    with st.expander(
        "View Patient Summary",
        expanded=True
    ):

        st.write(
            "**Patient:**",
            summary.get(
                "patient_name",
                "Not Mentioned"
            )
        )

        st.write(
            "**Diagnosis:**",
            summary.get(
                "diagnosis",
                "Not Mentioned"
            )
        )

        st.write(
            "**Cancer Stage:**",
            summary.get(
                "cancer_stage",
                "Not Mentioned"
            )
        )

        st.markdown("### 💊 Treatments")

        treatments = summary.get(
            "treatments",
            []
        )

        if treatments:

            for item in treatments:
                st.write(f"• {item}")

        else:
            st.write("Not Mentioned")

        st.markdown("### 💉 Medications")

        medications = summary.get(
            "medications",
            []
        )

        if medications:

            for item in medications:
                st.write(f"• {item}")

        else:
            st.write("Not Mentioned")

        st.markdown("### 📋 Recommendations")

        recommendations = summary.get(
            "recommendations",
            []
        )

        if recommendations:

            for item in recommendations:
                st.write(f"• {item}")

        else:
            st.write("Not Mentioned")

# ---------------------------------------------------
# AI CASE SUMMARY
# ---------------------------------------------------

case_summary = st.session_state.case_summary

if case_summary:

    st.subheader("📄 AI Cancer Case Summary")

    patient = case_summary.get(
        "patient_information",
        {}
    )

    with st.container(border=True):

        st.markdown("### 👤 Patient Information")

        st.write(
            f"**Name:** {patient.get('name','Not Mentioned')}"
        )

        st.write(
            f"**Age:** {patient.get('age','Not Mentioned')}"
        )

        st.write(
            f"**Gender:** {patient.get('gender','Not Mentioned')}"
        )

    sections = [

        ("🩺 Primary Diagnosis",
         "primary_diagnosis"),

        ("🧬 Cancer Type",
         "cancer_type"),

        ("📍 Cancer Site",
         "cancer_site"),

        ("📏 Tumor Size",
         "tumor_size"),

        ("📊 Cancer Stage",
         "cancer_stage"),

        ("🔬 Histopathology",
         "histopathology"),

        ("📖 Clinical History",
         "clinical_history"),

        ("📍 Current Status",
         "current_status"),

        ("📅 Follow-up",
         "follow_up"),

    ]

    for title, key in sections:

        with st.container(border=True):

            st.markdown(f"### {title}")

            st.write(
                case_summary.get(
                    key,
                    "Not Mentioned"
                )
            )

    with st.container(border=True):

        st.markdown("### 💊 Treatments")

        treatments = case_summary.get(
            "treatments",
            []
        )

        if treatments:

            for treatment in treatments:
                st.write(f"• {treatment}")

        else:
            st.write("Not Mentioned")

    with st.container(border=True):

        st.markdown("### 💉 Medications")

        medications = case_summary.get(
            "medications",
            []
        )

        if medications:

            for medication in medications:
                st.write(f"• {medication}")

        else:
            st.write("Not Mentioned")

    with st.container(border=True):

        st.markdown("### 📝 Recommendations")

        recommendations = case_summary.get(
            "recommendations",
            []
        )

        if recommendations:

            for recommendation in recommendations:
                st.write(f"• {recommendation}")

        else:
            st.write("Not Mentioned")

    findings = case_summary.get(
        "key_findings",
        []
    )

    if findings:

        st.markdown("### 🔍 Key Findings")

        for finding in findings:

            st.success(finding)

    # ---------------------------------------------------
# STRUCTURED MEDICAL INFORMATION
# ---------------------------------------------------

if st.session_state.reports:

    st.subheader("📋 Structured Medical Information")

    for report in st.session_state.reports:

        info = report["structured_info"]

        with st.expander(f"📄 {report['filename']}", expanded=False):

            col1, col2 = st.columns(2)

            with col1:

                st.markdown("### 👤 Patient")

                st.write("**Name:**", info.get("patient_name", "Not Mentioned"))
                st.write("**Age:**", info.get("age", "Not Mentioned"))
                st.write("**Gender:**", info.get("gender", "Not Mentioned"))
                st.write("**Report Date:**", info.get("report_date", "Not Mentioned"))
                st.write("**Report Type:**", info.get("report_type", "Not Mentioned"))

            with col2:

                st.markdown("### 🩺 Diagnosis")

                st.write("**Diagnosis:**", info.get("diagnosis", "Not Mentioned"))
                st.write("**Cancer Type:**", info.get("cancer_type", "Not Mentioned"))
                st.write("**Cancer Site:**", info.get("cancer_site", "Not Mentioned"))
                st.write("**Tumor Size:**", info.get("tumor_size", "Not Mentioned"))
                st.write("**Cancer Stage:**", info.get("cancer_stage", "Not Mentioned"))

            st.divider()

            st.markdown("### 📖 Clinical History")
            st.write(info.get("clinical_history", "Not Mentioned"))

            st.markdown("### 🔬 Findings")
            st.write(info.get("findings", "Not Mentioned"))

            st.markdown("### 📌 Impression")
            st.write(info.get("impression", "Not Mentioned"))

            st.markdown("### 💊 Treatment")

            treatments = info.get("treatments", [])

            if treatments:

                for treatment in treatments:
                    st.write(f"• {treatment}")

            else:
                st.write("Not Mentioned")

            st.markdown("### 💉 Medications")

            medications = info.get("medications", [])

            if medications:

                for medication in medications:
                    st.write(f"• {medication}")

            else:
                st.write("Not Mentioned")

            st.markdown("### 📅 Follow-up")
            st.write(info.get("follow_up", "Not Mentioned"))

            st.markdown("### 📝 Recommendations")

            recommendations = info.get("recommendations", [])

            if recommendations:

                for recommendation in recommendations:
                    st.write(f"• {recommendation}")

            else:
                st.write("Not Mentioned")

# ---------------------------------------------------
# MEDICAL TERMS
# ---------------------------------------------------

st.subheader("🩺 Medical Terms Explained")

all_terms_text = ""

for report in st.session_state.reports:

    terms = report["structured_info"].get(
        "medical_terms",
        []
    )

    if terms:

        all_terms_text += "\n".join(terms)
        all_terms_text += "\n"

if all_terms_text.strip():

    try:

        explanation = simplify_terms(
            all_terms_text
        )

        st.markdown(explanation)

    except Exception as e:

        st.error(str(e))

# ---------------------------------------------------
# AI REPORT EXPLANATION
# ---------------------------------------------------

st.subheader("📄 AI Report Explanation")

if st.button(
    "Generate AI Explanation",
    use_container_width=True
):

    with st.spinner("Analyzing reports..."):

        combined_text = ""

        for report in st.session_state.reports:
            combined_text += report["text"]
            combined_text += "\n\n"

        try:

            explanation = explain_report(
                combined_text
            )

            st.session_state.report_explanation = (
                explanation
            )

        except Exception as e:

            st.error(e)

if st.session_state.report_explanation:

    st.markdown(
        st.session_state.report_explanation
    )

# ---------------------------------------------------
# AI PATIENT SUMMARY
# ---------------------------------------------------

st.subheader("🧬 AI Patient Summary")

if st.button(
    "Generate Patient Summary",
    use_container_width=True
):

    with st.spinner("Generating summary..."):

        combined_text = ""

        for report in st.session_state.reports:

            combined_text += report["text"]
            combined_text += "\n\n"

        try:

            summary = generate_patient_summary(
                combined_text
            )

            st.session_state.patient_summary_text = (
                summary
            )

        except Exception as e:

            st.error(e)

if st.session_state.patient_summary_text:

    st.markdown(
        st.session_state.patient_summary_text
    )

# ---------------------------------------------------
# CHATBOT
# ---------------------------------------------------

st.subheader("💬 Ask Questions About Your Reports")

# Display chat history
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input(
    "Ask a question about your medical reports..."
)

if question:

    # -----------------------------------------
    # USER MESSAGE
    # -----------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    st.session_state.memory.add_message(
        "user",
        question
    )

    with st.chat_message("user"):
        st.markdown(question)

    # -----------------------------------------
    # ANSWER GENERATION
    # -----------------------------------------

    answer = ""
    retrieved_docs = []
    evidence = None

    if st.session_state.vector_db:

        try:

            context = retrieve_context(
                st.session_state.vector_db,
                question
            )

            answer, detected_report = answer_question(
                question=question,
                context=context,
                memory=st.session_state.memory,
                patient_summary=st.session_state.patient_summary
            )

            retrieved_docs = (
                st.session_state.vector_db
                .similarity_search(question, k=3)
            )

            evidence = analyze_evidence(
                question,
                [
                    doc.page_content
                    for doc in retrieved_docs
                ]
            )

        except Exception as e:

            answer = f"❌ Error\n\n{e}"

    else:

        answer = (
            "Please upload one or more medical reports first."
        )

    # -----------------------------------------
    # ASSISTANT RESPONSE
    # -----------------------------------------

    with st.chat_message("assistant"):

        st.markdown(answer)

        if evidence:

            st.divider()

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Evidence Quality",
                    evidence["quality"]
                )

            with col2:

                st.metric(
                    "Confidence",
                    evidence["confidence"]
                )

            st.info(evidence["reason"])

            if evidence.get("missing_info"):

                st.warning(
                    "Missing Information:\n\n"
                    + "\n".join(
                        f"• {item}"
                        for item in evidence["missing_info"]
                    )
                )

        # -------------------------------------
        # REPORT EVIDENCE
        # -------------------------------------

        if retrieved_docs:

            with st.expander(
                "📄 Supporting Report Evidence"
            ):

                for i, doc in enumerate(
                    retrieved_docs,
                    start=1
                ):

                    st.markdown(
                        f"### Evidence {i}"
                    )

                    st.write(
                        doc.page_content
                    )

                    st.divider()

    # -----------------------------------------
    # SAVE CHAT HISTORY
    # -----------------------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [
                d.page_content
                for d in retrieved_docs
            ]
        }
    )

    st.session_state.memory.add_message(
        "assistant",
        answer
    )

    st.session_state.memory.summarize()

    # ---------------------------------------------------
    # FOOTER
    # ---------------------------------------------------

    st.divider()

    st.caption(
        "🧬 OncoGuide AI • Cancer Report Understanding using LLMs, RAG and Streamlit"
    )