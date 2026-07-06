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


# -----------------------------
# UPLOAD REPORTS
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload Medical Reports",
    type=["pdf"],
    accept_multiple_files=True
)

if uploaded_files:

    for uploaded_file in uploaded_files:

        existing_files = [r["filename"] for r in st.session_state.reports]

        if uploaded_file.name in existing_files:
            continue

        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(uploaded_file.read())
            pdf_path = temp.name

        # Extract + classify
        text = extract_text_from_pdf(pdf_path)
        report_type = detect_report_type(text)

        # Extract structured medical information
        structured_info = extract_medical_information(
            text,
            report_type
        )

        st.session_state.reports.append({
            "filename": uploaded_file.name,
            "type": report_type,
            "text": text,
            "structured_info": structured_info
         })

    # -----------------------------
    # Combine all reports
    # -----------------------------
    all_text = ""

    for r in st.session_state.reports:

        structured_summary = r["structured_info"]["structured_summary"]

        all_text += f"""
        REPORT NAME:
            {r['filename']}

        REPORT TYPE:
            {r['type']}

        STRUCTURED MEDICAL SUMMARY:
            {structured_summary}

        ORIGINAL REPORT:
            {r['text']}

        ----------------------------
    """

    # -----------------------------
    # Create Vector DB (RAG)
    # -----------------------------
    chunks = chunk_text(all_text)
    embedding_model = get_embedding_model()

    st.session_state.vector_db = create_vector_store(
        chunks,
        embedding_model
    )

    st.success("Reports uploaded successfully")


    # -----------------------------
    # REPORT LIST
    # -----------------------------
    st.subheader("📚 Uploaded Reports")

    for r in st.session_state.reports:
        st.write(f"📄 {r['filename']} - {r['type']}")

    # -----------------------------
    # STRUCTURED MEDICAL INFORMATION
    # -----------------------------
    st.subheader("📋 Structured Medical Information")

    for report in st.session_state.reports:

        info = report["structured_info"]

        with st.expander(f"📄 {report['filename']}"):

            st.markdown(f"### 🩺 Diagnosis")
            st.write(info["diagnosis"])

            st.markdown("### 🔍 Findings")
            st.write(info["findings"])

            st.markdown("### 📌 Impression")
            st.write(info["impression"])

            st.markdown("### 💊 Treatment")
            st.write(info["treatment"])

            st.markdown("### 📅 Recommendations")
            st.write(info["recommendations"])

            st.markdown("### ⚠️ Abnormal Values")
            st.write(info["abnormal_values"])

            st.markdown("### 🧬 Medical Terms")

            if info["medical_terms"]:
                for term in info["medical_terms"]:
                    st.write(f"• {term}")
            else:
                st.write("None")


    # -----------------------------
    # MEDICAL TERMS
    # -----------------------------
    with st.expander("🩺 Medical Terms Explained"):
        terms = simplify_terms(all_text)
        st.write(terms if terms else "No medical terms found.")


    # -----------------------------
    # FULL REPORT VIEW
    # -----------------------------
    with st.expander("📄 Combined Report Content"):
        st.text(all_text)


    # -----------------------------
    # AI REPORT SUMMARY
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


# -----------------------------
# CHAT SECTION
# -----------------------------
st.subheader("💬 Ask Questions About Your Reports")


# Show chat history
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📄 Evidence From Report"):
                for s in msg["sources"]:
                    st.write(s)
                    st.divider()


# -----------------------------
# CHAT INPUT
# -----------------------------
question = st.chat_input("Ask about your reports...")

if question:

    # Save user message
    st.session_state.memory.add_message("user", question)

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

                # STEP 1: Retrieve context
                context = retrieve_context(
                    st.session_state.vector_db,
                    question
                )

                # STEP 2: Generate answer
                answer, detected_report = answer_question(
                    question,
                    context,
                    st.session_state.memory
                )

                # STEP 3: store raw docs for UI evidence
                results = st.session_state.vector_db.similarity_search(question, k=3)

        except Exception as e:
            answer = f"Error: {e}"

    else:
        answer = "Please upload medical reports first."

    # -----------------------------
    # SHOW ASSISTANT RESPONSE
    # -----------------------------
    with st.chat_message("assistant"):

        if detected_report:
            st.info(f"🧠 Detected Report: {detected_report.upper()}")
        else:
            st.info("🧠 Detected Report: ALL REPORTS")

        st.markdown(answer)

        if results:
            with st.expander("📄 Evidence From Reports"):
                for doc in results:
                    st.write(doc.page_content)
                    st.divider()

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": [d.page_content for d in results]
    })

    st.session_state.memory.add_message("assistant", answer)
    st.session_state.memory.summarize()