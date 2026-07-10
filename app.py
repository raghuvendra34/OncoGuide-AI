import tempfile
import traceback
import streamlit as st

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
from src.case_summary_generator import CaseSummaryGenerator

# NEW (Day 20)
from src.patient_journey import PatientJourneyGenerator

from services.retrieval_service import retrieve_context
from services.chat_service import answer_question

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="OncoGuide AI", layout="wide")

st.title("🧬 OncoGuide AI")
st.caption("Understand Your Cancer Report with AI")
st.divider()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
defaults = {
    "vector_db": None,
    "memory": ConversationMemory(),
    "messages": [],
    "reports": [],
    "patient_summary": {},
    "patient_journey": "",
    "patient_timeline": PatientTimeline(),
    "case_summary": {}
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --------------------------------------------------
# REPORT UPLOAD
# --------------------------------------------------
uploaded_files = st.file_uploader(
    "Upload Medical Reports",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    new_reports = []

    for uploaded_file in uploaded_files:

        existing = [r["filename"] for r in st.session_state.reports]

        if uploaded_file.name in existing:
            continue

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(uploaded_file.read())
            pdf_path = temp.name

        text = extract_text_from_pdf(pdf_path)
        report_type = detect_report_type(text)

        structured_info = extract_medical_information(
            text,
            report_type
        )

        structured_info["report_type"] = report_type

        report_data = {
            "filename": uploaded_file.name,
            "type": report_type,
            "text": text,
            "structured_info": structured_info
        }

        st.session_state.reports.append(report_data)
        new_reports.append(structured_info)

        st.session_state.patient_timeline.add_report(
            report_text=text,
            report_type=report_type
        )

    if new_reports:

        all_structured_reports = [
            report["structured_info"]
            for report in st.session_state.reports
        ]

        reasoner = CrossReportReasoner(all_structured_reports)
        st.session_state.patient_summary = reasoner.build_summary()
        
        case_generator = CaseSummaryGenerator()

        timeline = st.session_state.patient_timeline.get_timeline()

        st.session_state.case_summary = case_generator.generate_summary(
            extracted_information=st.session_state.patient_summary,
            patient_timeline=timeline
        )

        reports_for_journey = [
            {
                "filename": r["filename"],
                "report_type": r["type"],
                "text": r["text"]
            }
            for r in st.session_state.reports
        ]

        try:
            journey_generator = PatientJourneyGenerator()
            st.session_state.patient_journey = (
                journey_generator.generate_journey(
                    reports_for_journey
                )
            )
        except Exception:
            pass

        all_text = ""

        for report in st.session_state.reports:

            structured_summary = report["structured_info"].get(
                "structured_summary",
                ""
            )

            all_text += f"""
REPORT: {report['filename']}
TYPE: {report['type']}

STRUCTURED SUMMARY:
{structured_summary}

ORIGINAL REPORT:
{report['text']}

--------------------------------------------------
"""

        chunks = chunk_text(all_text)

        embedding_model = get_embedding_model()

        st.session_state.vector_db = create_vector_store(
            chunks,
            embedding_model
        )

        st.success("Reports uploaded successfully")

# --------------------------------------------------
# UPLOADED REPORTS
# --------------------------------------------------
if st.session_state.reports:

    st.subheader("📚 Uploaded Reports")

    for report in st.session_state.reports:
        st.write(
            f"📄 {report['filename']} - {report['type']}"
        )

    # --------------------------------------------------
    # TIMELINE
    # --------------------------------------------------
    st.subheader("🕒 Patient Medical Timeline")

    timeline = (
        st.session_state.patient_timeline.get_timeline()
    )

    if timeline:

        for event in timeline:

            st.markdown(
                f"""
### 📅 {event['date']}

**Event:**  
{event['event']}

**Report Type:**  
{event['report_type']}

---
"""
            )

    # --------------------------------------------------
    # PATIENT JOURNEY
    # --------------------------------------------------
    st.subheader("🩺 AI Patient Journey")

    if st.session_state.patient_journey:
        st.markdown(
            st.session_state.patient_journey
        )

    # --------------------------------------------------
    # PATIENT SUMMARY
    # --------------------------------------------------
    st.subheader("🧠 Cross Report Patient Summary")

    summary = st.session_state.patient_summary

    with st.expander(
        "View Patient Summary",
        expanded=True
    ):
        st.write("Diagnosis:", summary.get("diagnosis"))
        st.write("Cancer Stage:", summary.get("cancer_stage"))
        st.write("Treatments:", summary.get("treatments"))
        st.write("Medications:", summary.get("medications"))
        st.write("Biomarkers:", summary.get("biomarkers"))
        st.write("Tumor Details:", summary.get("tumor_details"))
        st.write("Lab Results:", summary.get("lab_results"))
        st.write(
            "Recommendations:",
            summary.get("recommendations")
        )
    # --------------------------------------------------
    # AI CASE SUMMARY
    # --------------------------------------------------
    st.subheader("📄 AI Cancer Case Summary")

    case_summary = st.session_state.case_summary

    if case_summary:
        
        sections = [
            ("👤 Patient Information", "patient_information"),
            ("🩺 Primary Diagnosis", "primary_diagnosis"),
            ("📊 Cancer Stage", "cancer_stage"),
            ("📈 Disease Progression", "disease_progression"),
            ("💊 Treatments", "treatments"),
            ("✅ Response To Treatment", "response_to_treatment"),
            ("📍 Current Status", "current_status"),
            ("📅 Follow-up", "follow_up"),
            ("📝 Clinical Summary", "clinical_summary"),
        ]

        for title, key in sections:
            with st.container(border=True):
                st.markdown(f"### {title}")
                st.write(case_summary.get(key, "Not Available"))

        if case_summary.get("key_findings"):
            st.markdown("### 🔍 Key Findings")

            for finding in case_summary["key_findings"]:
                st.success(finding)    

    # --------------------------------------------------
    # STRUCTURED INFO
    # --------------------------------------------------
    st.subheader("📋 Structured Medical Information")

    for report in st.session_state.reports:

        info = report["structured_info"]

        with st.expander(f"📄 {report['filename']}"):

            st.write("Diagnosis:", info.get("diagnosis"))
            st.write("Findings:", info.get("findings"))
            st.write("Impression:", info.get("impression"))
            st.write("Treatment:", info.get("treatment"))
            st.write(
                "Recommendations:",
                info.get("recommendations")
            )

    # --------------------------------------------------
    # MEDICAL TERMS
    # --------------------------------------------------
    with st.expander("🩺 Medical Terms Explained"):

        combined_text = "\\n".join(
            [r["text"] for r in st.session_state.reports]
        )

        st.write(
            simplify_terms(combined_text)
        )

    # --------------------------------------------------
    # AI REPORT EXPLANATION
    # --------------------------------------------------
    if st.button("Explain My Reports"):

        try:

            combined_text = "\\n".join(
                [r["text"] for r in st.session_state.reports]
            )

            with st.spinner("Analyzing reports..."):

                explanation = explain_report(
                    combined_text
                )

            st.subheader(
                "AI Report Explanation"
            )

            st.markdown(explanation)

        except Exception as e:

            st.error(str(e))
            st.code(traceback.format_exc())

# --------------------------------------------------
# CHATBOT
# --------------------------------------------------
st.subheader("💬 Ask Questions About Your Reports")

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):

        st.markdown(msg["content"])

question = st.chat_input(
    "Ask about your reports..."
)

if question:

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

    if st.session_state.vector_db:

        try:

            context = retrieve_context(
                st.session_state.vector_db,
                question
            )

            answer, detected_report = (
                answer_question(
                    question=question,
                    context=context,
                    memory=st.session_state.memory,
                    patient_summary=st.session_state.patient_summary
                )
            )

            results = (
                st.session_state.vector_db
                .similarity_search(question, k=3)
            )

        except Exception as e:

            answer = f"Error: {e}"
            results = []

    else:

        answer = (
            "Please upload medical reports first."
        )
        results = []

    with st.chat_message("assistant"):

        st.markdown(answer)

        if results:

            with st.expander(
                "📄 Evidence From Reports"
            ):

                for doc in results:
                    st.write(doc.page_content)
                    st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [
                d.page_content
                for d in results
            ]
        }
    )

    st.session_state.memory.add_message(
        "assistant",
        answer
    )

    st.session_state.memory.summarize()