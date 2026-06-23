from langchain_community.vectorstores import FAISS


def create_vector_store(chunks, embedding_model):

    db = FAISS.from_texts(
        chunks,
        embedding_model
    )

    return db