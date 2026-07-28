"""Step 4: store chunk embeddings in FAISS so we can search them later."""  

import os
from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_models


# build vector sstore

def build_vector_store(chunks):
    '''Embed every chunck and build
    a searchable FAISS index in memoery.'''
    embeddings_model = get_embeddings_models()
    return FAISS.from_documents(chunks, embeddings_model)

## save vector store 

def save_vector_store(vector_store, path: str = config.VECTOR_STORE_PATH)-> None:
    """Save the FAISS vector store to disk
    so we don't have to rebuid it every time."""
    vector_store.save_local(path)


def load_vector_store(path: str = config.VECTOR_STORE_PATH):

    """Load the FAISS vector store from disk."""
    embeddings_model = get_embeddings_models()
    return FAISS.load_local(path, embeddings_model, allow_dangerous_deserialization=True)


def vector_store_exists(path: str = config.VECTOR_STORE_PATH) -> bool:  
    """Check if the FAISS vector store exists on disk."""
    return os.path.exists(os.path.join(path, "index.faiss")) 


def get_retriever(vector_store, top_k: int = config.TOP_K_RESULTS):
    """Turn a vector store into a retriever
    that returns the top k results."""
    return vector_store.as_retriever(search_kwargs={"k": top_k})
    