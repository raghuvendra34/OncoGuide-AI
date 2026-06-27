import tempfile
import streamlit as st
import traceback

from src.chatbot import answer_question
from src.pdf_reader import extract_text_from_pdf
from src.text_chunker import chunk_text
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store
from src.medical_terms import simplify_terms
from src.report_explainer import explain_report

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="OncoGuide AI",
    layout="wide"
)

# -----------------------------
# Header
# -----------------------------
st.title("OncoGuide AI")
st.caption("Understand Your Cancer Report with AI")

st.divider()

# -----------------------------
# Session State
# -----------------------------
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "report_text" not in st.session_state:
    st.session_state.report_text = ""

if "messages" not in st.session_state:
    st.session_state.messages = []

# NEW: Conversation Memory
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Medical Report",
    type=["pdf"]
)

if uploaded_file is not None:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    # Extract text
    report_text = extract_text_from_pdf(temp_pdf_path)

    # Save report
    st.session_state.report_text = report_text

    # Create chunks
    chunks = chunk_text(report_text)

    # Embedding model
    embedding_model = get_embedding_model()

    # Vector Store
    vector_db = create_vector_store(
        chunks,
        embedding_model
    )

    st.session_state.vector_db = vector_db

    st.success("Report uploaded successfully.")

    st.divider()

    # -----------------------------
    # Medical Terms
    # -----------------------------
    with st.expander("Medical Terms Explained", expanded=False):

        explanations = simplify_terms(report_text)

        if explanations:
            st.write(explanations)
        else:
            st.write("No matching medical terms found.")

    # -----------------------------
    # Report Content
    # -----------------------------
    with st.expander("Report Content", expanded=False):
        st.text(report_text)

    # -----------------------------
    # AI Report Explanation
    # -----------------------------
    st.divider()

    if st.button("Explain My Report"):

        try:

            with st.spinner("Analyzing report..."):

                explanation = explain_report(report_text)

            st.subheader("AI Report Explanation")
            st.markdown(explanation)

        except Exception as e:

            st.error(f"Error generating explanation: {e}")
            st.code(traceback.format_exc())

# -----------------------------
# Chatbot
# -----------------------------
st.subheader("Ask Questions About Your Report")

# Display previous chat
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and "sources" in message
            and message["sources"]
        ):

            with st.expander("📄 Evidence from Your Report"):

                for i, source in enumerate(message["sources"], start=1):

                    st.markdown(f"**Evidence {i}**")

                    st.write(source)

                    st.divider()

# -----------------------------
# User Input
# -----------------------------
question = st.chat_input(
    "Ask a question about your report..."
)

if question:

    # Show user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    # -----------------------------
    # Generate Answer
    # -----------------------------
    if st.session_state.vector_db:

        try:

            with st.spinner("Thinking..."):

                results = st.session_state.vector_db.similarity_search(
                    question,
                    k=3
                )

            # Build context
            context = "\n\n".join(
                [doc.page_content for doc in results]
            )

            # Generate answer WITH conversation memory
            answer = answer_question(
                question=question,
                context=context,
                chat_history=st.session_state.chat_history
            )

        except Exception as e:

            answer = f"Error: {str(e)}"
            results = []

    else:

        answer = "Please upload a medical report first."
        results = []

    # -----------------------------
    # Display Assistant Response
    # -----------------------------
    with st.chat_message("assistant"):

        st.markdown(answer)

        if results:

            with st.expander("📄 Evidence from Your Report"):

                for i, doc in enumerate(results, start=1):

                    st.markdown(f"**Evidence {i}**")

                    st.write(doc.page_content)

                    st.divider()

    # Save UI messages
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": [doc.page_content for doc in results] if results else []
        }
    )

    # -----------------------------
    # Save Conversation Memory
    # -----------------------------
    st.session_state.chat_history.append(
        {
            "question": question,
            "answer": answer
        }
    )