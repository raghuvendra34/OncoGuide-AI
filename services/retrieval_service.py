from src.chunk_ranker import rank_chunks
from src.context_fusion import merge_chunks


def retrieve_context(vector_db, query, k=8):
    """
    Retrieve the most relevant report context from FAISS,
    rank the chunks by clinical importance,
    and merge them into a cleaner context for the LLM.
    """

    # Retrieve more chunks
    docs = vector_db.similarity_search(query, k=k)

    # Extract text
    chunks = [
        doc.page_content.strip()
        for doc in docs
        if doc.page_content.strip()
    ]

    # Rank by medical importance
    ranked_chunks = rank_chunks(chunks)

    # Merge top chunks
    merged_context = merge_chunks(
        ranked_chunks,
        max_chunks=5
    )

    return merged_context