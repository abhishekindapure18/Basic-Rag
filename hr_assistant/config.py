
import os
from dotenv import load_dotenv

load_dotenv()


## ENV VAR / SECRET

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

## DEFINE PATH - DATA / VECTORE STORE

DATA_FILE_PATH = os.path.join("data", "hr_policy.txt")

VECTOR_STORE_PATH = os.path.join("data", "faiss_index")

## MODELS
# LLM and EMBEDDING MODEL

LLM_MODEL_NAME = "openai/gpt-oss-20b"

EMBEDDING_MODEL_NAME = "jina-embeddings-v2-base-en"

## CHUNK / TEXT SPLITTING CONFIG

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# RETRIVAL RESULTS

TOP_K_RESULTS = 3

## SYSTEM INSSTRUCTIONS

system_prompt=(
    """
    You are a friendly HR assistant for Acme Corporation.
    Always use the search_hr_policy tool to look up facts before answering.
    if the answer isn't in the search results, say you don't know instead of guessing.
    """
)

def check_api_keys() -> None:
    """
    Stop early with a clear message if a reguired API key is missing.
    """
    if not GROQ_API_KEY:
        raise ValueError("Missing GROQ_API_KEY. Please add it to the environment variables.")
    if not JINA_API_KEY:
        raise ValueError("Missing JINA_API_KEY. Please add it to the environment variables.")

        