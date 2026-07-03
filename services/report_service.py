from src.pdf_reader import extract_text_from_pdf
from src.report_classifier import detect_report_type
from src.text_chunker import chunk_text


def process_report(file_path: str):
    """
    Extract, classify, and prepare a medical report.
    """

    # Step 1: Extract text from PDF
    text = extract_text_from_pdf(file_path)

    if not text:
        return {
            "text": "",
            "type": "Unknown",
            "chunks": []
        }

    # Step 2: Detect report type
    report_type = detect_report_type(text)

    # Step 3: Chunk text for RAG
    chunks = chunk_text(text)

    return {
        "text": text,
        "type": report_type,
        "chunks": chunks
    }