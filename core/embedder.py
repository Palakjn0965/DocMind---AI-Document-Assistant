import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def get_embeddings():
    """
    Returns a HuggingFace embedding model.
    Runs locally
    First run downloads the model (~90MB), cached after that.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embeddings


def create_vector_store(chunks):
    """
    Takes chunks from split_documents().
    Creates embeddings and stores them in FAISS.
    Returns the vector store object.
    """
    embeddings = get_embeddings()

    print("Creating embeddings... (first run downloads ~90MB model)")
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
    print("Vector store created successfully!")
    return vector_store


def save_vector_store(vector_store, path: str = "vector_store"):
    """
    Saves the vector store to disk.
    So you don't re-embed every time you restart the app.
    """
    vector_store.save_local(path)
    print(f"Vector store saved to '{path}' folder")


def load_vector_store(path: str = "vector_store"):
    """
    Loads a previously saved vector store from disk.
    """
    embeddings = get_embeddings()
    vector_store = FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
    print(f"Vector store loaded from '{path}'")
    return vector_store


def get_retriever(vector_store, k: int = 4):
    """
    Wraps vector store as a retriever.
    k = how many chunks to return per query.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever