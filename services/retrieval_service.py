def retrieve_context(vector_db, query, k=3):
    """
    PURE FAISS retrieval layer.
    """

    docs = vector_db.similarity_search(query, k=k)

    return "\n\n".join(
        f"SECTION {i+1}\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )