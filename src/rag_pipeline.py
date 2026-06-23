from src.pdf_reader import extract_text
from src.text_chunker import chunk_text
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store


def build_rag_pipeline(pdf_path):
    """
    Build the RAG pipeline from a PDF report.
    Returns a FAISS vector database.
    """

    # Extract text from PDF
    text = extract_text(pdf_path)

    # Split text into chunks
    chunks = chunk_text(text)

    # Load embedding model
    embedding_model = get_embedding_model()

    # Create vector database
    vector_db = create_vector_store(
        chunks,
        embedding_model
    )

    return vector_db


def retrieve_context(vector_db, query, k=3):
    """
    Retrieve the most relevant chunks for a query.
    """

    results = vector_db.similarity_search(
        query,
        k=k
    )

    context = "\n\n".join(
        [doc.page_content for doc in results]
    )

    return context