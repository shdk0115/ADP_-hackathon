import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_TEXT_MODEL = os.getenv("BEDROCK_TEXT_MODEL", "amazon.titan-text-lite-v1")
BEDROCK_EMBED_MODEL = os.getenv("BEDROCK_EMBED_MODEL", "amazon.titan-embed-text-v1")

OPENSEARCH_HOST = os.getenv("OPENSEARCH_HOST", "localhost")
OPENSEARCH_PORT = int(os.getenv("OPENSEARCH_PORT", "9200"))
OPENSEARCH_USE_SSL = os.getenv("OPENSEARCH_USE_SSL", "false").lower() == "true"
OPENSEARCH_USER = os.getenv("OPENSEARCH_USER")
OPENSEARCH_PASSWORD = os.getenv("OPENSEARCH_PASSWORD")

OPENSEARCH_DEFAULTS = {
    "number_of_shards": 1,
    "number_of_replicas": 1,
    "knn": True,
}

EMBEDDING_CONFIG = {
    "dimension": 768,
    "similarity": "cosine",
}

DEFAULT_INDEX_NAME = "docs_default_v1"
AGENT_INDEX_PREFIX = "docs_agent"
MAX_ANALYSIS_RETRY = 3
