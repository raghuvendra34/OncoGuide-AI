import tempfile
import traceback
import streamlit as st


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


if "report_text" not in st.session_state:
    st.session_state.report_text = ""


if "messages" not in st.session_state:
    st.session_state.messages = []


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []



# -----------------------------
# Upload Report
# -----------------------------

uploaded_file = st.file_uploader(
    "Upload Medical Report",
    type=["pdf"]
)


if uploaded_file:


    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp:


        temp.write(
            uploaded_file.read()
        )

        pdf_path = temp.name



    report_text = extract_text_from_pdf(
        pdf_path
    )


    st.session_state.report_text = report_text



    chunks = chunk_text(
        report_text
    )


    embedding_model = get_embedding_model()



    vector_db = create_vector_store(
        chunks,
        embedding_model
    )


    st.session_state.vector_db = vector_db


    st.success(
        "Report uploaded successfully"
    )


    # -----------------------------
    # Medical Terms
    # -----------------------------

    with st.expander(
        "🩺 Medical Terms Explained"
    ):


        terms = simplify_terms(
            report_text
        )


        if terms:
            st.write(terms)

        else:
            st.write(
                "No matching medical terms found."
            )



    # -----------------------------
    # Report Content
    # -----------------------------

    with st.expander(
        "📄 Report Content"
    ):

        st.text(
            report_text
        )



    # -----------------------------
    # AI Explanation
    # -----------------------------

    st.divider()


    if st.button(
        "Explain My Report"
    ):


        try:


            with st.spinner(
                "Analyzing report..."
            ):


                explanation = explain_report(
                    report_text
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
    "Ask Questions About Your Report"
)



# Previous Messages

for message in st.session_state.messages:


    with st.chat_message(
        message["role"]
    ):


        st.markdown(
            message["content"]
        )


        if (
            message["role"]=="assistant"
            and message.get("sources")
        ):


            with st.expander(
                "📄 Evidence From Report"
            ):


                st.markdown(
                    "### 📌 Most Relevant Section"
                )


                st.write(
                    message["sources"][0]
                )


                if len(message["sources"]) > 1:


                    st.divider()


                    st.markdown(
                        "### 📄 Supporting Sections"
                    )


                    for source in message["sources"][1:]:

                        st.write(
                            source
                        )

                        st.divider()



# -----------------------------
# User Question
# -----------------------------

question = st.chat_input(
    "Ask about your report..."
)



if question:


    st.session_state.messages.append(

        {
            "role":"user",
            "content":question
        }

    )


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )



    results=[]



    if st.session_state.vector_db:


        try:


            with st.spinner(
                "Searching report..."
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
                    for i,doc in enumerate(results)
                ]

            )



            answer = answer_question(

                question,

                context,

                st.session_state.chat_history

            )


        except Exception as e:


            answer = f"Error: {e}"



    else:


        answer = (
            "Please upload a medical report first."
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
                "📄 Evidence From Your Report"
            ):


                st.markdown(
                    "### 📌 Most Relevant Section"
                )


                st.write(
                    results[0].page_content
                )


                if len(results)>1:


                    st.divider()


                    st.markdown(
                        "### 📄 Supporting Sections"
                    )


                    for doc in results[1:]:

                        st.write(
                            doc.page_content
                        )

                        st.divider()




    # Save Message

    st.session_state.messages.append(

        {
            "role":"assistant",
            "content":answer,
            "sources":
            [
                doc.page_content
                for doc in results
            ]
        }

    )



    # Save Memory

    st.session_state.chat_history.append(

        {
            "question":question,
            "answer":answer
        }

    )