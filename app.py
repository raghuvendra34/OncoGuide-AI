import tempfile
import traceback
import streamlit as st

from src.report_selector import detect_requested_report
from src.report_classifier import detect_report_type
from src.memory import ConversationMemory
from src.pdf_reader import extract_text_from_pdf
from src.text_chunker import chunk_text
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.medical_terms import simplify_terms
from src.report_explainer import explain_report
from src.medical_information_extractor import extract_medical_information
from src.cross_report_reasoner import CrossReportReasoner
from src.patient_timeline import PatientTimeline

from services.retrieval_service import retrieve_context
from services.chat_service import answer_question


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="OncoGuide AI",
    layout="wide"
)

st.title("🧬 OncoGuide AI")
st.caption("Understand Your Cancer Report with AI")
st.divider()


# -----------------------------
# SESSION STATE
# -----------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "reports" not in st.session_state:
    st.session_state.reports = []

if "patient_summary" not in st.session_state:
    st.session_state.patient_summary = {}

if "patient_timeline" not in st.session_state:
    st.session_state.patient_timeline = PatientTimeline()


# -----------------------------
# UPLOAD REPORTS
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload Medical Reports",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    all_extracted_reports = []

    for uploaded_file in uploaded_files:

        existing_files = [
            r["filename"]
            for r in st.session_state.reports
        ]

        if uploaded_file.name in existing_files:
            continue

        # -----------------------------
        # Save Temporary PDF
        # -----------------------------
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp:

            temp.write(uploaded_file.read())
            pdf_path = temp.name

        # -----------------------------
        # Extract Text
        # -----------------------------
        text = extract_text_from_pdf(pdf_path)

        # -----------------------------
        # Detect Report Type
        # -----------------------------
        report_type = detect_report_type(text)
        st.session_state.patient_timeline.add_report(
            report_text=text,
            report_type=report_type
        )

        # -----------------------------
        # Structured Medical Extraction
        # -----------------------------
        structured_info = extract_medical_information(
            text,
            report_type
        )

        # Add report type for reasoning
        structured_info["report_type"] = report_type

        all_extracted_reports.append(structured_info)

        st.session_state.reports.append({

            "filename": uploaded_file.name,

            "type": report_type,

            "text": text,

            "structured_info": structured_info
        })

    # -----------------------------
    # DAY 18
    # Cross Report Reasoner
    # -----------------------------
    reasoner = CrossReportReasoner(
        all_extracted_reports
    )

    st.session_state.patient_summary = (
        reasoner.build_summary()
    )

    # -----------------------------
    # Combine Reports
    # -----------------------------
    all_text = ""

    for report in st.session_state.reports:

        structured_summary = report["structured_info"].get(
            "structured_summary",
            ""
        )

        all_text += f"""

REPORT NAME:
{report['filename']}

REPORT TYPE:
{report['type']}

STRUCTURED MEDICAL SUMMARY:

{structured_summary}

ORIGINAL REPORT:

{report['text']}

----------------------------------------

"""

    # -----------------------------
    # Build Vector Database
    # -----------------------------
    chunks = chunk_text(all_text)

    embedding_model = get_embedding_model()

    st.session_state.vector_db = create_vector_store(
        chunks,
        embedding_model
    )

    st.success("Reports uploaded successfully")

    # -----------------------------
    # Uploaded Reports
    # -----------------------------
    st.subheader("📚 Uploaded Reports")

    for report in st.session_state.reports:
        st.write(f"📄 {report['filename']} - {report['type']}")

    # -----------------------------
    # Patient Timeline
    # -----------------------------
    st.subheader("🕒 Patient Medical Timeline")

    timeline = st.session_state.patient_timeline.get_timeline()

    if timeline:

        for event in timeline:

            st.markdown(
                f"""
                ### 📅 {event['date']}

                **Event**

                {event['event']}

                **Report Type**

                {event['report_type']}

                ---
                """
                )

    else:

        st.info("Upload reports to generate a timeline.")

    # -----------------------------
    # Day 18
    # Patient Summary
    # -----------------------------
    st.subheader("🧠 Cross Report Patient Summary")

    summary = st.session_state.patient_summary

    with st.expander("View Patient Summary", expanded=True):

        st.markdown("### Diagnosis")
        st.write(summary.get("diagnosis", "Not available"))

        st.markdown("### Cancer Stage")
        st.write(summary.get("cancer_stage", "Not available"))

        st.markdown("### Treatments")
        treatments = summary.get("treatments", [])
        if treatments:
            for t in treatments:
                st.write(f"• {t}")
        else:
            st.write("No treatments extracted.")

        st.markdown("### Medications")
        medications = summary.get("medications", [])
        if medications:
            for m in medications:
                st.write(f"• {m}")
        else:
            st.write("No medications extracted.")

        st.markdown("### Biomarkers")
        biomarkers = summary.get("biomarkers", {})
        if biomarkers:
            st.json(biomarkers)
        else:
            st.write("None")

        st.markdown("### Tumor Details")
        tumor = summary.get("tumor_details", {})
        if tumor:
            st.json(tumor)
        else:
            st.write("None")

        st.markdown("### Lab Results")
        labs = summary.get("lab_results", {})
        if labs:
            st.json(labs)
        else:
            st.write("None")

        st.markdown("### Recommendations")
        recommendations = summary.get("recommendations", [])
        if recommendations:
            for r in recommendations:
                st.write(f"• {r}")
        else:
            st.write("None")

    # -----------------------------
    # Structured Medical Information
    # -----------------------------
    st.subheader("📋 Structured Medical Information")

    for report in st.session_state.reports:

        info = report["structured_info"]

        with st.expander(f"📄 {report['filename']}"):

            st.markdown("### 🩺 Diagnosis")
            st.write(info.get("diagnosis"))

            st.markdown("### 🔍 Findings")
            st.write(info.get("findings"))

            st.markdown("### 📌 Impression")
            st.write(info.get("impression"))

            st.markdown("### 💊 Treatment")
            st.write(info.get("treatment"))

            st.markdown("### 📅 Recommendations")
            st.write(info.get("recommendations"))

            st.markdown("### ⚠️ Abnormal Values")
            st.write(info.get("abnormal_values"))

            st.markdown("### 🧬 Medical Terms")

            medical_terms = info.get("medical_terms", [])

            if medical_terms:
                for term in medical_terms:
                    st.write(f"• {term}")
            else:
                st.write("None")

    # -----------------------------
    # Medical Terms
    # -----------------------------
    with st.expander("🩺 Medical Terms Explained"):

        terms = simplify_terms(all_text)

        st.write(
            terms if terms else "No medical terms found."
        )

    # -----------------------------
    # Full Report
    # -----------------------------
    with st.expander("📄 Combined Report Content"):
        st.text(all_text)

    # -----------------------------
    # AI Report Explanation
    # -----------------------------
    st.divider()

    if st.button("Explain My Reports"):

        try:

            with st.spinner("Analyzing reports..."):

                explanation = explain_report(all_text)

            st.subheader("AI Report Explanation")

            st.markdown(explanation)

        except Exception as e:

            st.error(str(e))

            st.code(traceback.format_exc())


# =====================================================
# CHAT SECTION
# =====================================================

st.subheader("💬 Ask Questions About Your Reports")

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):

            with st.expander("📄 Evidence From Reports"):

                for source in msg["sources"]:

                    st.write(source)

                    st.divider()


question = st.chat_input("Ask about your reports...")

if question:

    st.session_state.memory.add_message(
        "user",
        question
    )

    st.session_state.messages.append({

        "role": "user",

        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    answer = ""

    results = []

    detected_report = None

    if st.session_state.vector_db:

        try:

            with st.spinner("Searching reports..."):

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

                results = st.session_state.vector_db.similarity_search(
                    question,
                    k=3
                )

        except Exception as e:

            answer = f"Error: {e}"

    else:

        answer = "Please upload medical reports first."

    with st.chat_message("assistant"):

        if detected_report:

            st.info(
                f"🧠 Detected Report: {detected_report.upper()}"
            )

        else:

            st.info("🧠 Detected Report: ALL REPORTS")

        st.markdown(answer)

        if results:

            with st.expander("📄 Evidence From Reports"):

                for doc in results:

                    st.write(doc.page_content)

                    st.divider()

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "sources": [
            d.page_content for d in results
        ]
    })

    st.session_state.memory.add_message(
        "assistant",
        answer
    )

    st.session_state.memory.summarize()