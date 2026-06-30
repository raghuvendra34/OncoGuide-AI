import tempfile
import traceback
import streamlit as st

from src.report_classifier import detect_report_type
from src.memory import ConversationMemory
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

st.title("🧬 OncoGuide AI")
st.caption(
    "Understand Your Cancer Report with AI"
)

st.divider()


# -----------------------------
# Session State
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
# Upload Multiple Reports
# -----------------------------

uploaded_files = st.file_uploader(
    "Upload Medical Reports",
    type=["pdf"],
    accept_multiple_files=True
)


if uploaded_files:


    for uploaded_file in uploaded_files:


        # Avoid duplicate processing

        existing_files = [
            report["filename"]
            for report in st.session_state.reports
        ]


        if uploaded_file.name in existing_files:
            continue



        # Save temporary PDF

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp:


            temp.write(
                uploaded_file.read()
            )


            pdf_path = temp.name



        # Extract text

        report_text = extract_text_from_pdf(
            pdf_path
        )


        # Detect report type

        report_type = detect_report_type(
            report_text
        )



        # Store report

        report_data = {

            "filename": uploaded_file.name,

            "type": report_type,

            "text": report_text

        }


        st.session_state.reports.append(
            report_data
        )



    # -----------------------------
    # Combine Reports
    # -----------------------------

    all_reports_text = ""


    for report in st.session_state.reports:


        all_reports_text += f"""

REPORT NAME:
{report['filename']}


REPORT TYPE:
{report['type']}


CONTENT:

{report['text']}


----------------------------

"""



    # -----------------------------
    # Create Vector Database
    # -----------------------------

    chunks = chunk_text(
        all_reports_text
    )


    embedding_model = get_embedding_model()


    vector_db = create_vector_store(
        chunks,
        embedding_model
    )


    st.session_state.vector_db = vector_db



    st.success(
        "Reports uploaded successfully"
    )



    # -----------------------------
    # Report List
    # -----------------------------

    st.subheader(
        "📚 Uploaded Reports"
    )


    for report in st.session_state.reports:


        st.write(
            f"📄 {report['filename']} - {report['type']}"
        )



    # -----------------------------
    # Medical Terms
    # -----------------------------

    with st.expander(
        "🩺 Medical Terms Explained"
    ):


        terms = simplify_terms(
            all_reports_text
        )


        if terms:

            st.write(
                terms
            )


        else:

            st.write(
                "No matching medical terms found."
            )



    # -----------------------------
    # Report Content
    # -----------------------------

    with st.expander(
        "📄 Combined Report Content"
    ):


        st.text(
            all_reports_text
        )



    # -----------------------------
    # AI Explanation
    # -----------------------------

    st.divider()


    if st.button(
        "Explain My Reports",
        key="explain_reports"
    ):


        try:


            with st.spinner(
                "Analyzing reports..."
            ):


                explanation = explain_report(
                    all_reports_text
                )



            st.subheader(
                "AI Report Explanation"
            )


            st.markdown(
                explanation
            )



        except Exception as e:


            st.error(
                str(e)
            )


            st.code(
                traceback.format_exc()
            )




# -----------------------------
# Chat Section
# -----------------------------

st.subheader(
    "Ask Questions About Your Reports"
)



# -----------------------------
# Show Previous Messages
# -----------------------------

for message in st.session_state.messages:


    with st.chat_message(
        message["role"]
    ):


        st.markdown(
            message["content"]
        )


        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):


            with st.expander(
                "📄 Evidence From Report"
            ):


                st.write(
                    message["sources"][0]
                )




# -----------------------------
# Chat Input
# -----------------------------

question = st.chat_input(
    "Ask about your reports..."
)



if question:



    st.session_state.memory.add_message(
        "user",
        question
    )



    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )



    with st.chat_message(
        "user"
    ):


        st.markdown(
            question
        )



    results = []



    if st.session_state.vector_db:



        try:


            with st.spinner(
                "Searching reports..."
            ):


                results = (

                    st.session_state.vector_db
                    .similarity_search(
                        question,
                        k=3
                    )

                )



            context = "\n\n".join(

                [

                    f"SECTION {i+1}\n{doc.page_content}"

                    for i, doc in enumerate(results)

                ]

            )



            answer = answer_question(

                question,

                context,

                st.session_state.memory

            )



        except Exception as e:


            answer = f"Error: {e}"



    else:


        answer = (
            "Please upload medical reports first."
        )




    # -----------------------------
    # Display Answer
    # -----------------------------

    with st.chat_message(
        "assistant"
    ):


        st.markdown(
            answer
        )


        if results:


            with st.expander(
                "📄 Evidence From Reports"
            ):


                for doc in results:


                    st.write(
                        doc.page_content
                    )

                    st.divider()




    # Save assistant response

    st.session_state.messages.append(

        {

            "role": "assistant",

            "content": answer,

            "sources":

            [

                doc.page_content

                for doc in results

            ]

        }

    )



    st.session_state.memory.add_message(

        "assistant",

        answer

    )


    st.session_state.memory.summarize()