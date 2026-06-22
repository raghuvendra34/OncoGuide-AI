import streamlit as st

from src.pdf_reader import extract_text_from_pdf
from src.report_explainer import explain_report

st.set_page_config(
    page_title="OncoGuide AI",
    page_icon="🩺"
)

st.title("🩺 OncoGuide AI")
st.write("Upload your cancer report and get a simple explanation.")

uploaded_file = st.file_uploader(
    "Upload PDF Report",
    type=["pdf"]
)

if uploaded_file:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    report_text = extract_text_from_pdf("temp.pdf")

    st.subheader("Extracted Report Text")

    with st.expander("View Text"):
        st.write(report_text)

    if st.button("Explain Report"):

        with st.spinner("Analyzing report..."):

            explanation = explain_report(report_text)

        st.subheader("AI Explanation")

        st.write(explanation)