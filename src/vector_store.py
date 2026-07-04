from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model, metadatas=None):
    """
    Create a FAISS vector database.

    Parameters
    ----------
    chunks : list[str]
        Text chunks extracted from the uploaded reports.

    embedding_model :
        Embedding model used to convert text into vectors.

    metadatas : list[dict], optional
        Metadata for each chunk (e.g., report type, file name).
    """

    if metadatas is None:
        db = FAISS.from_texts(
            chunks,
            embedding_model
        )
    else:
        db = FAISS.from_texts(
            texts=chunks,
            embedding=embedding_model,
            metadatas=metadatas
        )

    return db