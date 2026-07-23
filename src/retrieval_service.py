"""
retrieval_service.py

Handles retrieval of the most relevant report chunks from the
vector database along with their similarity scores.
"""


def retrieve_context(vector_db, query, k=3):
    """
    Retrieve relevant chunks from the vector database.

    Returns
    -------
    {
        "context": str,
        "chunks": [
            {
                "content": str,
                "score": float | None,
                "metadata": dict
            }
        ]
    }
    """

    try:
        # Preferred retrieval (includes similarity score)
        docs_and_scores = vector_db.similarity_search_with_score(query, k=k)

        chunks = []
        context_parts = []

        for doc, score in docs_and_scores:
            chunks.append(
                {
                    "content": doc.page_content,
                    "score": float(score),
                    "metadata": doc.metadata,
                }
            )

            context_parts.append(doc.page_content)

        return {
            "context": "\n\n".join(context_parts),
            "chunks": chunks,
        }

    except Exception:
        # Fallback if similarity scores are unavailable
        docs = vector_db.similarity_search(query, k=k)

        chunks = []

        for doc in docs:
            chunks.append(
                {
                    "content": doc.page_content,
                    "score": None,
                    "metadata": doc.metadata,
                }
            )

        return {
            "context": "\n\n".join(doc.page_content for doc in docs),
            "chunks": chunks,
        }