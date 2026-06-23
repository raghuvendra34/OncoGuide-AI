from src.pdf_reader import extract_text_from_pdf
from src.text_chunker import chunk_text
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_store


pdf_path = "data/reports/sample_report.pdf"


text = extract_text_from_pdf(pdf_path)

chunks = chunk_text(text)

embedding_model = get_embedding_model()

vector_db = create_vector_store(
    chunks,
    embedding_model
)


query = "What is the diagnosis?"

results = vector_db.similarity_search(query)


for r in results:
    print(r.page_content)