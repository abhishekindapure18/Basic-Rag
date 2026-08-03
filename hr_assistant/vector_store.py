"""Step 4: store chunk embeddings in FAISS so we can search them later."""  

import os
from langchain_community.vectorstores import FAISS
from hr_assistant import config
from hr_assistant.embeddings import get_embeddings_models
from hr_assistant.logger import get_logger

logger = get_logger(__name__)


# build vector sstore

def build_vector_store(chunks):
    '''Embed every chunck and build
    a searchable FAISS index in memoery.'''
    logger.info("Embedding %d chunk(s) and building FAISS index...", len(chunks))
    embeddings_model = get_embeddings_model()
    vector_store = FAISS.from_documents(chunks, embeddings_model)
    logger.info("FAISS index built in memory")
    return FAISS.from_documents(chunks, embeddings_model)

## save vector store 

def save_vector_store(vector_store, path: str = config.VECTOR_STORE_PATH)-> None:
    """Save the FAISS vector store to disk
    so we don't have to rebuid it every time."""
    vector_store.save_local(path)
    logger.info("Saved FAISS index to '%s'", path)


def load_vector_store(path: str = config.VECTOR_STORE_PATH):

    """Load the FAISS vector store from disk."""
    logger.info("Loading FAISS index from '%s'", path)
    embeddings_model = get_embeddings_models()
    return FAISS.load_local(path, embeddings_model, allow_dangerous_deserialization=True)


def vector_store_exists(path: str = config.VECTOR_STORE_PATH) -> bool:  
    """Check if the FAISS vector store exists on disk."""
    logger.info("Checking if FAISS index exists at '%s'", path)
    return os.path.exists(os.path.join(path, "index.faiss")) 


def get_retriever(vector_store, top_k: int = config.TOP_K_RESULTS):
    """Turn a vector store into a retriever
    that returns the top k results."""
    logger.info("Creating retriever with top_k=%d", top_k)
    return vector_store.as_retriever(search_kwargs={"k": top_k})
    