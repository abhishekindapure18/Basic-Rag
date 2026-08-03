"""Step 3: turn text into numbers (vectors) using Jina embeddings."""

from langchain_community.embeddings import JinaEmbeddings
from hr_assistant import config
from hr_assistant.logger import get_logger

logger = get_logger(__name__)

def get_embeddings_models():
    """Return a Jina embeddings model.
    Reads Jina API key from environment variable JINA_API_KEY."""
    logger.info("Initializing embeddings model '%s'", config.EMBEDDING_MODEL_NAME)
    return JinaEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
    )